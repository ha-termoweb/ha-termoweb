/*
 * cores3_display.h - optional CoreS3 SE screen for the TermoWeb radio gateway.
 *
 * Separate from termoweb_radio.h on purpose: this file is a local addition, not
 * part of the upstream gateway. It plugs into the generic hook points of
 * termoweb_radio.h (set_hooks) and shares the radio's SPI bus.
 *
 * Starts with a condensed view of all heaters stacked, each on its own background,
 * then one heater per page (name, temperature, set point, heating state). The
 * background turns orange while a heater is heating. next_page() steps through the
 * views; the YAML calls it when the touch screen is tapped. Text is anti-aliased
 * (cores3_font.h, made by gen_font.py) and composed row by row by cores3_gfx.h.
 *
 * Write-only, and drawn from the radio task while the receiver is idle, so it
 * can never overlap a CC1101 access. The screen's DC line is the same pad as the
 * CC1101's MISO: it is an output only while a drawing call runs and goes back to
 * a plain input before the radio touches the bus.
 *
 * YAML (on_boot, priority -100, before termoweb::radio.start()):
 *   cores3::display.install(3, 35);   // chip select GPIO3, data/command GPIO35
 * and from Home Assistant sensors:
 *   cores3::display.next_page(); set_invert(bool)
 * The integration pushes each heater with the gateway's 'L' command.
 */
#pragma once

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "cores3_gfx.h"
#include "termoweb_radio.h"

namespace cores3 {

class Display {
 public:
  static constexpr int MAX_PAGES = 4;

  /* Registers the screen with the radio. Call before termoweb::radio.start(). */
  void install(int cs, int dc) {
    this->cs_ = cs;
    this->dc_ = dc;
    termoweb::TermowebRadio::Hooks h;
    self_ = this;
    h.setup = [](spi_host_device_t host) { self_->setup_(host); };
    h.work_pending = []() { return self_->pending_(); };
    h.work = []() { self_->draw_(); };
    h.line = [](const char *text) { self_->handle_line(text); };
    termoweb::radio.set_hooks(h);
  }

  /* Thread-safe setters: remember the value; the radio task draws it when idle. */
  void set_name(int page, const std::string &name) {
    if (!valid_(page)) return;
    portENTER_CRITICAL(&this->mux_);
    snprintf(this->pages_[page].name, sizeof(this->pages_[page].name), "%s", name.c_str());
    portEXIT_CRITICAL(&this->mux_);
    this->dirty_ = true;
  }
  void set_temp(int page, float v) { this->set_(page, &Page::temp, v); }
  void set_setpoint(int page, float v) { this->set_(page, &Page::setpoint, v); }
  void set_power(int page, float v) { this->set_(page, &Page::power, v); }
  void set_action(int page, const std::string &action) {
    if (!valid_(page)) return;
    portENTER_CRITICAL(&this->mux_);
    this->pages_[page].action_heating = action == "heating";
    portEXIT_CRITICAL(&this->mux_);
    this->dirty_ = true;
  }
  /* One heater from the integration: "page|count|name|temp|setpoint|power|heating". */
  void handle_line(const char *text) {
    char buf[96];
    snprintf(buf, sizeof(buf), "%s", text);
    char *f[7];
    int n = 0;
    for (char *p = buf; n < 7;) {
      f[n++] = p;
      p = strchr(p, '|');
      if (p == nullptr) break;
      *p++ = '\0';
    }
    if (n != 7) return;
    int page = atoi(f[0]), count = atoi(f[1]);
    if (!valid_(page) || count < 1 || count > MAX_PAGES) return;
    portENTER_CRITICAL(&this->mux_);
    Page &pg = this->pages_[page];
    snprintf(pg.name, sizeof(pg.name), "%s", f[2]);
    pg.temp = strtof(f[3], nullptr);
    pg.setpoint = strtof(f[4], nullptr);
    pg.power = strtof(f[5], nullptr);
    pg.action_heating = f[6][0] == '1';
    this->npages_ = count;
    if (this->page_ >= count) this->page_ = 0;
    if (count < 2) this->overview_ = false;
    portEXIT_CRITICAL(&this->mux_);
    this->dirty_ = true;
  }
  /* Tap order: all heaters together, then each heater full screen, then back. */
  void next_page() {
    if (this->overview_) {
      this->overview_ = false;
      this->page_ = 0;
    } else if (this->page_ + 1 < this->npages_) {
      this->page_ = this->page_ + 1;
    } else if (this->npages_ > 1) {
      this->overview_ = true;
    } else {
      this->page_ = 0;
    }
    this->dirty_ = true;
  }
  void set_invert(bool on) {
    this->invert_ = on ? 1 : 0;
    this->invert_dirty_ = true;
    this->dirty_ = true;
  }

  /* Adds the panel as a second device on the radio's SPI bus and brings it up. */
  bool attach(spi_host_device_t host) {
    spi_device_interface_config_t dev = {};
    dev.mode = 0;
    dev.clock_speed_hz = 20000000;
    dev.spics_io_num = this->cs_;
    dev.queue_size = 1;
    if (spi_bus_add_device(host, &dev, &this->dev_) != ESP_OK) return false;
    this->begin_();
    this->cmd_(0x01);
    vTaskDelay(pdMS_TO_TICKS(150));
    this->cmd_(0x11);
    vTaskDelay(pdMS_TO_TICKS(120));
    this->cmd_(0x3A, 0x55); /* RGB565 */
    this->cmd_(0x36, 0x08); /* landscape, BGR */
    this->cmd_(this->invert_ ? 0x21 : 0x20);
    this->invert_dirty_ = false;
    this->cmd_(0x29);
    this->fill_black_();
    this->end_();
    this->ready_ = true;
    return true;
  }

 private:
  bool valid_(int page) const { return page >= 0 && page < MAX_PAGES; }
  void set_(int page, float Page::*field, float v) {
    if (!valid_(page)) return;
    portENTER_CRITICAL(&this->mux_);
    this->pages_[page].*field = v;
    portEXIT_CRITICAL(&this->mux_);
    this->dirty_ = true;
  }
  void setup_(spi_host_device_t host) {
    if (this->attach(host)) this->draw_();
  }
  bool pending_() { return this->ready_ && this->dirty_.exchange(false); }

  /* Redraws only the horizontal bands whose content changed. */
  void draw_() {
    Page pgs[MAX_PAGES];
    const int cur = this->page_;
    portENTER_CRITICAL(&this->mux_);
    for (int i = 0; i < MAX_PAGES; i++) pgs[i] = this->pages_[i];
    portEXIT_CRITICAL(&this->mux_);
    Screen s;
    if (this->overview_ && this->npages_ > 1) layout_overview(pgs, this->npages_, s);
    else layout(pgs[cur], cur, this->npages_, s);
    this->begin_();
    if (this->invert_dirty_.exchange(false)) this->cmd_(this->invert_ ? 0x21 : 0x20);
    uint8_t row[SCREEN_W * 3];
    for (int b = MAX_BANDS - 1; b >= s.n_bands; b--) this->last_sig_[b][0] = '\0';
    for (int b = 0; b < s.n_bands; b++) {
      if (strcmp(s.band[b].sig, this->last_sig_[b]) == 0) continue;
      this->window_(0, s.band[b].y0, SCREEN_W, s.band[b].y1 - s.band[b].y0);
      for (int y = s.band[b].y0; y < s.band[b].y1; y++) {
        compose_row(s.band[b], y, cur, this->npages_, row);
        for (int x = 0; x < SCREEN_W; x++) {
          const uint8_t *p = row + x * 3;
          this->push_(((p[0] & 0xF8) << 8) | ((p[1] & 0xFC) << 3) | (p[2] >> 3));
        }
      }
      this->flush_();
      memcpy(this->last_sig_[b], s.band[b].sig, sizeof(this->last_sig_[b]));
    }
    this->end_();
  }
  void fill_black_() {
    this->window_(0, 0, SCREEN_W, SCREEN_H);
    for (long i = 0; i < (long) SCREEN_W * SCREEN_H; i++) this->push_(0);
    this->flush_();
    memset(this->last_sig_, 0, sizeof(this->last_sig_));
  }
  void begin_() {
    spi_device_acquire_bus(this->dev_, portMAX_DELAY);
    gpio_set_direction((gpio_num_t) this->dc_, GPIO_MODE_INPUT_OUTPUT);
  }
  void end_() {
    gpio_set_direction((gpio_num_t) this->dc_, GPIO_MODE_INPUT);
    spi_device_release_bus(this->dev_);
  }
  void xfer_(const uint8_t *p, size_t len, int dc) {
    gpio_set_level((gpio_num_t) this->dc_, dc);
    while (len > 0) {
      size_t chunk = len > 64 ? 64 : len;
      spi_transaction_t t;
      memset(&t, 0, sizeof(t));
      t.length = chunk * 8;
      t.tx_buffer = p;
      spi_device_polling_transmit(this->dev_, &t);
      p += chunk;
      len -= chunk;
    }
  }
  void cmd_(uint8_t c) { this->xfer_(&c, 1, 0); }
  void cmd_(uint8_t c, uint8_t d) {
    this->xfer_(&c, 1, 0);
    this->xfer_(&d, 1, 1);
  }
  void window_(int x, int y, int w, int h) {
    uint8_t a[4] = {(uint8_t) (x >> 8), (uint8_t) x, (uint8_t) ((x + w - 1) >> 8), (uint8_t) (x + w - 1)};
    uint8_t b[4] = {(uint8_t) (y >> 8), (uint8_t) y, (uint8_t) ((y + h - 1) >> 8), (uint8_t) (y + h - 1)};
    this->cmd_(0x2A);
    this->xfer_(a, 4, 1);
    this->cmd_(0x2B);
    this->xfer_(b, 4, 1);
    this->cmd_(0x2C);
  }
  void push_(uint16_t c) {
    this->buf_[this->n_++] = c >> 8;
    this->buf_[this->n_++] = c & 0xFF;
    if (this->n_ >= sizeof(this->buf_)) this->flush_();
  }
  void flush_() {
    if (this->n_ > 0) this->xfer_(this->buf_, this->n_, 1);
    this->n_ = 0;
  }

  static inline Display *self_{nullptr};
  int cs_{-1}, dc_{-1}, npages_{1};
  std::atomic<bool> dirty_{true}, invert_dirty_{false};
  std::atomic<int> page_{0}, invert_{1};
  std::atomic<bool> overview_{true};
  portMUX_TYPE mux_ = portMUX_INITIALIZER_UNLOCKED;
  Page pages_[MAX_PAGES];
  bool ready_{false};
  spi_device_handle_t dev_{nullptr};
  uint8_t buf_[64];
  size_t n_{0};
  char last_sig_[MAX_BANDS][160]{};
};

inline Display display;

}  // namespace cores3
