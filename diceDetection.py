
from pylab import *

import skimage
from skimage.color import *
from skimage import io
from skimage.feature import blob_doh
from sklearn.cluster import DBSCAN
import numpy as np

NUM_SAMPLES = 13

for i in range(1, NUM_SAMPLES + 1):
    slika = io.imread('dices' + str(i) + '.jpg')  # Učitavanje slike

    slika_hsv = rgb2hsv(slika)  # Iz RGB sistema u HSV
    slika_sat = slika_hsv[:, :, 1]  # Izdvajamo saturaciju jer od nje zavisi boja


    slika_bin = slika_sat > 0.5  # binarizacija

    slika_bin = skimage.morphology.opening(slika_bin, skimage.morphology.disk(3))  # otvaranje binarizovane slike

    slika_bin = skimage.morphology.erosion(slika_bin, skimage.morphology.disk(1))  # erozija binarizovane slike


    tackice = blob_doh(slika_bin, min_sigma=5, max_sigma=15, threshold=.05)  # trazimo tacke

    tackice_koordinate = (tackice.T[0:2]).T  # izdvajamo iz matrice "tackice" 1. i 2. kolonu
    # jer se u njima nalaze x i y koordinate tacaka

    pripadnost_tacaka = DBSCAN(eps=35, min_samples=1).fit(tackice_koordinate)  # primena density based skeniranja na
    # niz pronadjenih tacaka kako bismo nasli koja tacka pripada kojoj kockici na osnovu blizine tacaka jedna drugoj

    broj_kockica = max(pripadnost_tacaka.labels_) + 1  # kako dbscan funkcija broji od 0 do n-1, gde je n broj kockica,
    # broj kockica ce biti jednak najvecoj vrednosti koja funkcija dodeli nekoj tacki plus 1

    broj_na_kockici = []
    centri_kockica = []

    for i in range(broj_kockica):
        tackice_ite_kocke = tackice_koordinate[pripadnost_tacaka.labels_ == i]  # za svaku kockicu trazimo one tacke
        # koje pripadaju njoj(iz dbscan) i pravimo matricu sa koordinatama tih tackica svake kocke

        broj_na_kockici.append(len(tackice_ite_kocke))  # odredjujemo koliko svaka kockica ima tacaka

        centar_ite_kocke = np.mean(tackice_ite_kocke, axis=0)  # nalazimo srednju vrednost svih tacaka koje pripadaju
        # kockici sto nam predstavlja centar kocke

        centri_kockica.append(centar_ite_kocke)  # formiramo matricu centara svih kockica

    centri_kockica = np.array(centri_kockica)

    figura, axes = plt.subplots(figsize=(10, 10))

    for i in range(len(tackice)):
        Drawing_colored_circle = plt.Circle((tackice[i, 1], tackice[i, 0]), tackice[i, 2], fill=True, color='magenta')
        axes.add_artist(Drawing_colored_circle)

    for i in range(len(centri_kockica)):
        Drawing_colored_circle2 = plt.Circle((centri_kockica[i, 1], centri_kockica[i, 0]), 35, fill=False, color='red')
        axes.add_artist(Drawing_colored_circle2)

    axes.set_aspect(1)
    axes.add_artist(Drawing_colored_circle)
    axes.add_artist(Drawing_colored_circle2)

    for i in range(len(centri_kockica)):
        plt.text(20, 50 + i * 30, "kockica {0}:  {1}".format(i + 1, broj_na_kockici[i]), fontsize=20)

    axis('off')
    axes.imshow(slika)
    plt.title('Kockice')
    plt.show()