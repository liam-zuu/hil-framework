"""tools/s06_stream.py, the run manifest (Codex Sol round 3, S56, S58, S63): marks paired by run id, E lines
counted by their run field, a loss seen is a failure even when another mark is missing."""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import s06_stream as stm  # noqa: E402


def write(tmp_path, lines):
    p = tmp_path / 'ev.txt'
    p.write_text(''.join(f'{i + 1}\t{x}\n' for i, x in enumerate(lines)))
    return str(p)


START = '#run_start,run={r},first_data_seq=1,timestamp_us={t},qseq={q}'
END = '#run_end,run={r},last_data_seq=9,timestamp_us={t},ev_qdrop_run=0,enc_ovf_run=0,qseq={q}'
STATS = '#run_stats,run={r},log_send_err_run=0,enc_reads_run={n},qseq={q}'


def two_runs(tmp_path, stats_a=True):
    a = [START.format(r=1, t=100, q=1), END.format(r=1, t=200, q=2)]
    if stats_a:
        a.append(STATS.format(r=1, n=3000, q=3))
    return write(tmp_path, a + [START.format(r=2, t=300, q=4), END.format(r=2, t=400, q=5),
                                STATS.format(r=2, n=2999, q=6)])


def test_file_with_two_runs_needs_a_choice(tmp_path):
    m = stm.run_marks(two_runs(tmp_path))
    assert m['_multi'] and m['_runs'] == ['1', '2'] and 'run_start' not in m
    s = stm.enc_stream([1, 2], [150, 151], 0, 1000, m, runs=[1, 1])
    assert s['multi_run'] and not s['known']


def test_stats_of_the_next_run_are_not_taken(tmp_path):
    # Codex S58 counter-example: stats of run 1 lost, run 2 read 2999 times: run 1 must not borrow them
    m = stm.run_marks(two_runs(tmp_path, stats_a=False), run=1)
    assert m['_run'] == 1 and 'run_stats' not in m
    s = stm.enc_stream(list(range(1, 3000)), [100 + i for i in range(2999)], 0, 1e9, m, runs=[1] * 2999)
    assert not s['known']


def test_e_lines_counted_by_run_field(tmp_path):
    m = stm.run_marks(two_runs(tmp_path), run=2)
    assert m['run_stats']['kv']['enc_reads_run'] == '2999'
    eseq = list(range(1, 3000 + 2999 + 1))
    tb = [float(i) for i in range(len(eseq))]
    runs = [1] * 3000 + [2] * 2999
    s = stm.enc_stream(eseq, tb, 3000.0, 1e9, m, runs=runs)
    assert s['known'] and s['ok'] and s['run_reads_received'] == 2999
    # first line of run 2 lost: the lines of run 1 do not fill in for it
    s = stm.enc_stream(eseq[:3000] + eseq[3001:], tb[:3000] + tb[3001:], 3000.0, 1e9, m,
                       runs=runs[:3000] + runs[3001:])
    assert not s['ok'] and s['run_reads_missing'] == 1


def test_old_firmware_without_run_field_still_reads(tmp_path):
    p = write(tmp_path, ['#run_start,timestamp_us=100,qseq=1', '#run_end,timestamp_us=200,ev_qdrop_run=0,'
                         'enc_ovf_run=0,qseq=2', '#run_stats,enc_reads_run=2,qseq=3'])
    m = stm.run_marks(p)
    assert m['_run'] is None and m['run_stats']['kv']['enc_reads_run'] == '2'
    s = stm.enc_stream([1, 2], [150.0, 151.0], 0, 1e9, m)
    assert s['known'] and s['ok']


def test_loss_counter_fails_even_with_a_mark_missing(tmp_path):
    # S63: #run_end says a queue dropped lines; #run_stats is gone: a failure, not only "not proven"
    p = write(tmp_path, ['#run_start,timestamp_us=100,qseq=1',
                         '#run_end,timestamp_us=200,ev_qdrop_run=2,enc_ovf_run=0,qseq=2'])
    m = stm.run_marks(p)
    s = stm.enc_stream([1, 2], [150.0, 151.0], 0, 1e9, m)
    assert not s['ok'] and not s['known']
    q = stm.qseq_stream(p, m)
    assert not q['ok'] and q['known']


def test_qseq_hole_fails_without_the_drop_field(tmp_path):
    # S63: START and END there, qseq 2 missing, END without ev_qdrop_run: the hole still fails
    p = write(tmp_path, ['#run_start,timestamp_us=100,qseq=1', '#run_end,timestamp_us=200,enc_ovf_run=0,qseq=3'])
    q = stm.qseq_stream(p)
    assert q['missing'] == 1 and not q['ok'] and not q['known']


def test_line_with_qseq_but_no_payload_is_not_received(tmp_path):
    p = write(tmp_path, ['#run_start,timestamp_us=100,qseq=1', 'P,2', 'P,3,1,2,0,30,3,970,8,1000,1', 'S,4,10,1,0,0,0,0,0,'
                         '10000,0,0,0,0,0,nan', '#run_end,timestamp_us=200,ev_qdrop_run=0,enc_ovf_run=0,qseq=5'])
    q = stm.qseq_stream(p)
    assert q['malformed'] == 1 and q['missing'] == 1 and q['received'] == 4 and not q['ok']


def test_parse_e_accepts_both_formats():
    assert stm.parse_e('E,7,10,12,1,2,3,4'.split(',')) == (7, 10, 12, [1, 2, 3, 4], None)
    assert stm.parse_e('E,7,10,12,1,2,3,4,5'.split(',')) == (7, 10, 12, [1, 2, 3, 4], 5)
    assert stm.parse_e('E,7'.split(',')) is None
