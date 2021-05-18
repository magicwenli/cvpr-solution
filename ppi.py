from math import sqrt, pi

phone = [
    {"size": 4.0, "ph": 640, "pw": 480},
    {"size": 4.3, "ph": 640, "pw": 480},
    {"size": 4.0, "ph": 1136, "pw": 640},
    {"size": 5.0, "ph": 1920, "pw": 1080},
    {"size": 4.7, "ph": 1920, "pw": 1080},
]


def ppi(s, h, w):
    hw = h / w
    b2 = s ** 2 / (1 + hw ** 2)
    return w / sqrt(b2)


def getPpi(phone):
    for p in phone:
        print(ppi(p["size"], p["ph"], p["pw"]))


def getSpacing():
    for l in [0.5, 1, 2, 3]:
        print('{}m -> {:.3f}mm'.format(l, l * pi / 10.8))


def statisticsPixels(m, n, p):
    c = 0
    for x in range(m):
        for y in range(n):
            if p[m][n] > 5 and p[m][n] < 12:
                c += 1
    print(c)
