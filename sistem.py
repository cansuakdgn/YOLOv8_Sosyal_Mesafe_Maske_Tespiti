import cv2
import numpy as np
from ultralytics import YOLO
from math import dist
from datetime import datetime
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

def klasor_olustur(yol):
    if not os.path.exists(yol):
        os.makedirs(yol)

def perspektif_nokta(video):
    # Kullanıcıdan fare ile 4 adet perspektif noktası alır
    secilen_noktalar = []
    hedef_noktalar = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    secim_tamam = True

    def clik(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(secilen_noktalar) < 4:
            secilen_noktalar.append([x, y])
            print(f"{len(secilen_noktalar)}. nokta seçildi: ({x}, {y})")
            if len(secilen_noktalar) == 4:
                nonlocal secim_tamam
                secim_tamam = False

    Camera = cv2.VideoCapture(video)
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", clik)

    while secim_tamam:
        read, frame = Camera.read()
        if not read:
            Camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        kare = frame.copy()
        h, w = kare.shape[:2]

        for nokta in secilen_noktalar:
            cv2.circle(kare, tuple(nokta), 8, (0, 0, 255), -1)

        cv2.rectangle(kare, (0, h - 23), (w, h), (255, 255, 255), -1)
        cv2.putText(kare, "!!!   Zemin perspektifi icin 4 nokta secin: Sol-ust > Sag-ust > Sag-alt > Sol-alt   !!!",
                    (140, h - 7), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 1)
        cv2.imshow("Video", kare)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()

    Camera.release()
    cv2.destroyAllWindows()

    donusum_matrix = cv2.getPerspectiveTransform(np.float32(secilen_noktalar), hedef_noktalar)
    maskeleme_alani = np.array([secilen_noktalar], dtype=np.int32)
    return donusum_matrix, maskeleme_alani


def analiz(kare, model, takipci, donusum, maske_alani, kayitli_id, ihlal_kaydi):
    # Model ile tahmin yapar, kişileri takip eder ve ihlalleri tespit edip kareye çizer
    sonuc = model(kare)
    kutular = sonuc[0].boxes
    etiketler = model.names

    tespit = []
    bilgi_listesi = []

    for kutu in kutular:
        sinif = int(kutu.cls[0])
        etiket = etiketler[sinif]
        x1, y1, x2, y2 = map(int, kutu.xyxy[0])
        merkez_x, merkez_y = (x1 + x2) // 2, y2
        guven = float(kutu.conf[0])

        if cv2.pointPolygonTest(maske_alani, (merkez_x, merkez_y), False) < 0:
            continue

        tespit.append(([x1, y1, x2 - x1, y2 - y1], guven, etiket))
        bilgi_listesi.append((x1, y1, x2, y2, merkez_x, merkez_y, etiket))

    izlenen= takipci.update_tracks(tespit, frame=kare)
    izlenen_kisi= []

    for iz in izlenen:
        if not iz.is_confirmed():
            continue

        takip_id = iz.track_id
        etiket = iz.get_det_class()
        x1, y1, x2, y2 = map(int, iz.to_ltrb())
        cx, cy = (x1 + x2) // 2, y2

        ayak_noktasi = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), donusum)[0][0]

        if takip_id not in kayitli_id:
            kisi = kare[y1:y2, x1:x2]
            ss = f"screenshots/id_{takip_id}.jpg"
            cv2.imwrite(ss, kisi)
            kayitli_id.add(takip_id)

        izlenen_kisi.append({
            "id": takip_id,
            "kutu": (x1, y1, x2, y2),
            "ayak": ayak_noktasi,
            "etiket": etiket
        })

    ihlalli_id = set()
    for i in range(len(izlenen_kisi)):
        for j in range(i + 1, len(izlenen_kisi)):
            mesafe = dist(izlenen_kisi[i]["ayak"], izlenen_kisi[j]["ayak"])
            if mesafe < 150:
                ihlalli_id.add(izlenen_kisi[i]["id"])
                ihlalli_id.add(izlenen_kisi[j]["id"])

    for kisi in izlenen_kisi:
        pid = kisi["id"]
        x1, y1, x2, y2 = kisi["kutu"]
        etiket = kisi["etiket"]
        maske_ihlali = etiket == "no-mask"
        mesafe_ihlali = pid in ihlalli_id

        if maske_ihlali or mesafe_ihlali:
            if pid not in ihlal_kaydi:
                ihlal_kaydi[pid] = {
                    "mask": maske_ihlali,
                    "distance": mesafe_ihlali,
                    "time": datetime.now().strftime("%H:%M:%S")
                }

        if maske_ihlali and mesafe_ihlali:
            renk = (255, 0, 255)
        elif maske_ihlali:
            renk = (0, 0, 255)
        elif mesafe_ihlali:
            renk = (255, 0, 0)
        else:
            continue

        cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(kare, f"ID:{pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, renk, 1)
        if maske_ihlali:
            cv2.putText(kare, "Maske Yok", (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, renk, 1)

    # legend
    l_x, l_y = 10, 10
    space = 25
    w, h = 210, 85
    cv2.rectangle(kare, (l_x - 5, l_y - 5), (l_x + w, l_y + h), (0, 0, 0), -1)

    cv2.circle(kare, (l_x + 10, l_y + 15), 7, (0, 0, 255), -1)
    cv2.putText(kare, "Maske yok", (l_x + 25, l_y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.circle(kare, (l_x + 10, l_y + 15 + space), 7, (255, 0, 0), -1)
    cv2.putText(kare, "Mesafe ihlali", (l_x + 25, l_y + 20 + space), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.circle(kare, (l_x + 10, l_y + 15 + 2 * space), 7, (255, 0, 255), -1)
    cv2.putText(kare, "Maske ve mesafe yok", (l_x + 25, l_y + 20 + 2 * space), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    return kare


def ihlal_raporu(ihlal_kaydi):
    # İhlal yapanları konsola yazar
    maske_yok = [str(pid) for pid, v in ihlal_kaydi.items() if v["mask"] and not v["distance"]]
    mesafe_yok = [str(pid) for pid, v in ihlal_kaydi.items() if v["distance"] and not v["mask"]]
    her_iki = [f"ID {pid} - {v['time']}" for pid, v in ihlal_kaydi.items() if v["mask"] and v["distance"]]

    print("\n  GÜNCEL İHLAL RAPORU")
    print("---------------------------")
    print(f"Maske Takmayanlar: {', '.join(maske_yok) if maske_yok else 'YOK'}")
    print(f"Sosyal Mesafeyi İhlal Edenler: {', '.join(mesafe_yok) if mesafe_yok else 'YOK'}")
    print("Her İki Kurala Uymayanlar:")
    if her_iki:
        for kisi in her_iki:
            print(f"   - {kisi}")
    else:
        print("   - Yok")


def islem(video, model):
    # Tüm süreci başlatan ana fonksiyon
    try:
        print("[İNFO] Sistem baslatiliyor...")
        klasor_olustur("screenshots")

        model = YOLO(model)
        takip = DeepSort(max_age=30)
        donusum, maske_alani = perspektif_nokta(video)

        Camera = cv2.VideoCapture(video)
        ihlal_kaydi = {}
        kayitli_idler = set()

        while Camera.isOpened():
            read, frame = Camera.read()
            if not read:
                break

            frame = analiz(frame, model, takip, donusum, maske_alani, kayitli_idler, ihlal_kaydi)
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        ihlal_raporu(ihlal_kaydi)

    except Exception as hata:
        print(f"[HATA] {hata}")
    finally:
        Camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    islem(video="video.mp4", model="trained_mask_model.pt")
