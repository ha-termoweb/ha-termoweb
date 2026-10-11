/* rc_hal.h: what a platform must provide for the portable radio core.
 *
 * The core (radio_core.c, cc1101.c) never touches hardware directly. Each
 * port implements these functions once: firmware/nanocul/ for the AVR
 * nanoCUL, tests/firmware/host_hal.c for the host tests (a fake CC1101), and
 * later the ESP32 port (esphome/termoweb_radio.h).
 *
 * Calling context: rc_on_gdo2()/rc_on_gdo0() may run in an interrupt (AVR)
 * or in a task (ESP32). Everything else runs in the platform's main loop.
 */
#ifndef RC_HAL_H
#define RC_HAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Output: one byte to the host (UART / TCP). Lines end "\r\n"; the core
 * writes both bytes itself. May block (AVR UART) or buffer (ESP32). */
void rc_hal_putc(char c);

/* Free-running 32-bit microsecond clock; only differences are used. */
uint32_t rc_hal_micros(void);

/* Busy-wait (the CC1101 transmit refill loop depends on it). us <= 16000. */
void rc_hal_delay_us(uint16_t us);

/* CC1101 over SPI, register level. The platform owns CS and its
 * "wait for MISO low after CS low" rule (CC1101 datasheet 10.1). */
void rc_hal_chip_wake(void); /* CS low, wait for MISO low, CS high, 40 us */
uint8_t rc_hal_strobe(uint8_t cmd);
void rc_hal_write_reg(uint8_t addr, uint8_t val);
void rc_hal_write_burst(uint8_t addr, const uint8_t *buf, uint8_t len);
uint8_t rc_hal_read_status(uint8_t addr); /* burst bit set, as status registers need */
void rc_hal_read_burst(uint8_t addr, uint8_t *buf, uint8_t len);

/* GDO edge handling. mask: no rc_on_gdo0/2 calls from now on. unmask:
 * forget any edge latched while masked (our own transmission causes some),
 * then deliver edges again. */
void rc_hal_gdo_mask(void);
void rc_hal_gdo_unmask(void);

/* Short critical section against rc_on_gdo0/2 (AVR: cli/sei; a port whose
 * GDO handlers run in the same task as the main loop: no-op). */
void rc_hal_lock(void);
void rc_hal_unlock(void);

/* Raw survey ('R'): secs > 0 starts one (the platform sets rc.survey_active
 * and prints "# survey on ..."), secs == 0 ends the running one early. */
void rc_hal_survey(uint16_t secs);

#ifdef __cplusplus
}
#endif

#endif
