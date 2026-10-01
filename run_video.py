import cv2
import tomllib
from pathlib import Path

with Path(__file__).with_name("videos.toml").open("rb") as file:
    config = tomllib.load(file)

video = f"/home/joseph/reva_hard_examples/videos/{config['selected']}.mp4"
cap = cv2.VideoCapture(video)
delay = max(1, round(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30)))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Video", frame)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
