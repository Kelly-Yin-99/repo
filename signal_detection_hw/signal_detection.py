import matplotlib.pyplot as plt
from scipy.stats import norm


class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        vals = [hits, misses, false_alarms, correct_rejections]
        for v in vals:
            if not isinstance(v, (int, float)):
                raise ValueError("must be number")
            if v < 0:
                raise ValueError("cannot be negative")

        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def hit_rate(self):
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def false_alarm_rate(self):
        total = self.false_alarms + self.correct_rejections
        if total == 0:
            return 0.0
        return self.false_alarms / total

    def d_prime(self):
        h = self.hit_rate()
        fa = self.false_alarm_rate()

        # avoid inf
        if h == 0: h = 0.0001
        if h == 1: h = 0.9999
        if fa == 0: fa = 0.0001
        if fa == 1: fa = 0.9999
        return norm.ppf(h) - norm.ppf(fa)

    def criterion(self):
        h = self.hit_rate()
        fa = self.false_alarm_rate()

        if h == 0: h = 0.0001
        if h == 1: h = 0.9999
        if fa == 0: fa = 0.0001
        if fa == 1: fa = 0.9999

        return -0.5 * (norm.ppf(h) + norm.ppf(fa))

    def __add__(self, other):
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_alarms + other.false_alarms,
            self.correct_rejections + other.correct_rejections
        )

    def __sub__(self, other):
        return SignalDetection(
            self.hits - other.hits,
            self.misses - other.misses,
            self.false_alarms - other.false_alarms,
            self.correct_rejections - other.correct_rejections
        )

    def __mul__(self, factor):
        return SignalDetection(
            self.hits * factor,
            self.misses * factor,
            self.false_alarms * factor,
            self.correct_rejections * factor
        )

    def plot_sdt(self):
        d = self.d_prime()
        c = self.criterion()

        xs = []
        noise = []
        signal = []

        x = -4
        while x <= d + 4:
            xs.append(x)
            noise.append(norm.pdf(x, 0, 1))
            signal.append(norm.pdf(x, d, 1))
            x += 0.01

        fig, ax = plt.subplots()
        ax.plot(xs, noise, label="noise")
        ax.plot(xs, signal, label="signal")
        ax.axvline(c, linestyle="--", label="criterion")

        ax.set_title("SDT Plot")
        ax.legend()

        return fig, ax

    @staticmethod
    def plot_roc(sdt_list):
        pts = []
        for s in sdt_list:
            h = s.hit_rate()
            fa = s.false_alarm_rate()
            pts.append((h, fa))

        pts.sort()
        xs = [0]
        ys = [0]
        for h, fa in pts:
            xs.append(h)
            ys.append(fa)
        xs.append(1)
        ys.append(1)

        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker="o", label="ROC Curve")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")

        ax.set_xlabel("Hit Rate")
        ax.set_ylabel("False Alarm Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        return fig, ax


if __name__ == "__main__":

    sdts = [
        SignalDetection(95, 5, 60, 40),
        SignalDetection(90, 10, 50, 50),
        SignalDetection(85, 15, 40, 60),
        SignalDetection(80, 20, 30, 70),
        SignalDetection(70, 30, 20, 80),
        SignalDetection(60, 40, 15, 85),
        SignalDetection(50, 50, 10, 90),
        SignalDetection(40, 60, 5, 95),
    ]

    # ROC plot
    fig1, ax1 = SignalDetection.plot_roc(sdts)
    # SDT plot
    fig2, ax2 = sdts[1].plot_sdt()

    plt.show()