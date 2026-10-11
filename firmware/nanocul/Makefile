# termoweb_rx for the nanoCUL868 (Arduino Nano / ATmega328P + CC1101).
#
#   make                     build termoweb_rx.hex and print its size
#   make flash PORT=/dev/ttyUSB0
#   make signature PORT=...  find the bootloader's baud rate (57600 or 115200)
#   make size / make clean
#
# Needs avr-gcc, avr-libc, binutils-avr and avrdude
# (Debian/Ubuntu: apt install gcc-avr avr-libc binutils-avr avrdude).

MCU      = atmega328p
F_CPU    = 16000000UL
PORT     = /dev/ttyUSB0
# Optiboot (most current Nanos). The classic bootloader needs BAUD=57600.
BAUD     = 115200

# Line protocol 3.7, nanoCUL port (see firmware/README.md "Versions").
VERSION  = 3.7-nanocul
# PATABLE: 0xC0 = +10 dBm at 868 MHz (the level heaters were proven to ack).
PA       = 0xC0

CORE     = ../radio_core
CC       = avr-gcc
OBJCOPY  = avr-objcopy
SIZE     = avr-size
AVRDUDE  = avrdude

CFLAGS   = -mmcu=$(MCU) -DF_CPU=$(F_CPU) -DRC_VERSION=\"$(VERSION)\" -DRC_PA=$(PA) \
           -I$(CORE) -Os -std=gnu99 -Wall -Wextra -ffunction-sections -fdata-sections
LDFLAGS  = -mmcu=$(MCU) -Wl,--gc-sections

SRC      = main.c $(CORE)/radio_core.c $(CORE)/cc1101.c
HDR      = $(CORE)/radio_core.h $(CORE)/cc1101.h $(CORE)/rc_hal.h
OBJ      = main.o radio_core.o cc1101.o
TARGET   = termoweb_rx

.PHONY: all build size flash signature clean

all build: $(TARGET).hex size

$(TARGET).elf: $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) -o $@

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

main.o: main.c $(HDR) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: $(CORE)/%.c $(HDR) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

# Flash budget 30720 bytes (32 KB minus the 2 KB bootloader), RAM 2048.
size: $(TARGET).elf
	@echo "termoweb_rx $(VERSION), PA=$(PA)"
	$(SIZE) --format=avr --mcu=$(MCU) $(TARGET).elf || $(SIZE) $(TARGET).elf

signature:
	@echo "trying 57600 (classic bootloader)..."
	@$(AVRDUDE) -c arduino -p $(MCU) -P $(PORT) -b 57600 2>&1 | grep -q "device signature" \
		&& echo "57600 OK: flash with BAUD=57600" && exit 0; \
	echo "57600 failed, trying 115200 (Optiboot)..."; \
	$(AVRDUDE) -c arduino -p $(MCU) -P $(PORT) -b 115200

flash: $(TARGET).hex
	$(AVRDUDE) -c arduino -p $(MCU) -P $(PORT) -b $(BAUD) -D -U flash:w:$(TARGET).hex:i

clean:
	rm -f *.o *.elf *.hex
