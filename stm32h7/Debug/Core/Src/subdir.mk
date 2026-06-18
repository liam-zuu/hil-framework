################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/dma.c \
../Core/Src/encoder_gen.c \
../Core/Src/gpio.c \
../Core/Src/main.c \
../Core/Src/pwm_capture.c \
../Core/Src/spi.c \
../Core/Src/spi1_handler.c \
../Core/Src/stm32h7xx_hal_msp.c \
../Core/Src/stm32h7xx_it.c \
../Core/Src/syscalls.c \
../Core/Src/sysmem.c \
../Core/Src/system_stm32h7xx.c \
../Core/Src/tim.c \
../Core/Src/usart.c 

OBJS += \
./Core/Src/dma.o \
./Core/Src/encoder_gen.o \
./Core/Src/gpio.o \
./Core/Src/main.o \
./Core/Src/pwm_capture.o \
./Core/Src/spi.o \
./Core/Src/spi1_handler.o \
./Core/Src/stm32h7xx_hal_msp.o \
./Core/Src/stm32h7xx_it.o \
./Core/Src/syscalls.o \
./Core/Src/sysmem.o \
./Core/Src/system_stm32h7xx.o \
./Core/Src/tim.o \
./Core/Src/usart.o 

C_DEPS += \
./Core/Src/dma.d \
./Core/Src/encoder_gen.d \
./Core/Src/gpio.d \
./Core/Src/main.d \
./Core/Src/pwm_capture.d \
./Core/Src/spi.d \
./Core/Src/spi1_handler.d \
./Core/Src/stm32h7xx_hal_msp.d \
./Core/Src/stm32h7xx_it.d \
./Core/Src/syscalls.d \
./Core/Src/sysmem.d \
./Core/Src/system_stm32h7xx.d \
./Core/Src/tim.d \
./Core/Src/usart.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/%.o Core/Src/%.su Core/Src/%.cyclo: ../Core/Src/%.c Core/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DDEBUG -DUSE_PWR_LDO_SUPPLY -DUSE_HAL_DRIVER -DSTM32H753xx -c -I../Core/Inc -IC:/Users/liam/STM32Cube/Repository/STM32Cube_FW_H7_V1.13.0/Drivers/STM32H7xx_HAL_Driver/Inc -IC:/Users/liam/STM32Cube/Repository/STM32Cube_FW_H7_V1.13.0/Drivers/STM32H7xx_HAL_Driver/Inc/Legacy -IC:/Users/liam/STM32Cube/Repository/STM32Cube_FW_H7_V1.13.0/Drivers/BSP/STM32H7xx_Nucleo -IC:/Users/liam/STM32Cube/Repository/STM32Cube_FW_H7_V1.13.0/Drivers/CMSIS/Device/ST/STM32H7xx/Include -IC:/Users/liam/STM32Cube/Repository/STM32Cube_FW_H7_V1.13.0/Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src

clean-Core-2f-Src:
	-$(RM) ./Core/Src/dma.cyclo ./Core/Src/dma.d ./Core/Src/dma.o ./Core/Src/dma.su ./Core/Src/encoder_gen.cyclo ./Core/Src/encoder_gen.d ./Core/Src/encoder_gen.o ./Core/Src/encoder_gen.su ./Core/Src/gpio.cyclo ./Core/Src/gpio.d ./Core/Src/gpio.o ./Core/Src/gpio.su ./Core/Src/main.cyclo ./Core/Src/main.d ./Core/Src/main.o ./Core/Src/main.su ./Core/Src/pwm_capture.cyclo ./Core/Src/pwm_capture.d ./Core/Src/pwm_capture.o ./Core/Src/pwm_capture.su ./Core/Src/spi.cyclo ./Core/Src/spi.d ./Core/Src/spi.o ./Core/Src/spi.su ./Core/Src/spi1_handler.cyclo ./Core/Src/spi1_handler.d ./Core/Src/spi1_handler.o ./Core/Src/spi1_handler.su ./Core/Src/stm32h7xx_hal_msp.cyclo ./Core/Src/stm32h7xx_hal_msp.d ./Core/Src/stm32h7xx_hal_msp.o ./Core/Src/stm32h7xx_hal_msp.su ./Core/Src/stm32h7xx_it.cyclo ./Core/Src/stm32h7xx_it.d ./Core/Src/stm32h7xx_it.o ./Core/Src/stm32h7xx_it.su ./Core/Src/syscalls.cyclo ./Core/Src/syscalls.d ./Core/Src/syscalls.o ./Core/Src/syscalls.su ./Core/Src/sysmem.cyclo ./Core/Src/sysmem.d ./Core/Src/sysmem.o ./Core/Src/sysmem.su ./Core/Src/system_stm32h7xx.cyclo ./Core/Src/system_stm32h7xx.d ./Core/Src/system_stm32h7xx.o ./Core/Src/system_stm32h7xx.su ./Core/Src/tim.cyclo ./Core/Src/tim.d ./Core/Src/tim.o ./Core/Src/tim.su ./Core/Src/usart.cyclo ./Core/Src/usart.d ./Core/Src/usart.o ./Core/Src/usart.su

.PHONY: clean-Core-2f-Src

