import jetson_emulator.inference as inference

class JetsonNanoProfile:
    def __init__(self, power_mode="5W"):
        self.power_mode = power_mode
        self.net = None

    def load_imagenet(self, network="googlenet"):
        self.net = inference.imageNet(network)
        return self.net

    def set_power_mode(self, power_mode):
        self.power_mode = power_mode

    def get_specs(self):
        return {
            "name": "Jetson Nano (emulated)",
            "power_mode": self.power_mode
        }
