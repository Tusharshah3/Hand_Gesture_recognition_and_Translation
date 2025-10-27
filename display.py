from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from time import sleep
from PIL import ImageFont

class OledDisplay:
    def __init__(self):
        serial = i2c(port=1, address=0x3C)
        self.device = ssd1306(serial, width=128, height=32)
        self.font = ImageFont.load_default()

    def show_text(self, text):
        with canvas(self.device) as draw:
            draw.text((10, 10), text, fill="white", font=self.font)

    def startup_animation(self, title="TARANG"):
        for i in range(0, 128, 8):
            with canvas(self.device) as draw:
                draw.rectangle((0, 0, i, 32), outline="white", fill="white")
            sleep(0.05)
        self.show_text(title)
        sleep(1)
        self.show_text("System Booting...")
        sleep(1)
