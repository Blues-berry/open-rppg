import time
import cv2
import rppg

model = rppg.Model()

with model.video_capture(0):
    last_time = 0
    current_hr = None

    for frame, box in model.preview:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        now = time.time()
        if now - last_time > 1:
            result = model.hr(start=-10)
            if result and result.get("hr"):
                current_hr = result["hr"]
                print(f"HR: {current_hr:.1f} BPM")
            last_time = now

        if box is not None:
            y1, y2 = box[0]
            x1, x2 = box[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if current_hr is not None:
            cv2.putText(
                frame,
                f"HR: {current_hr:.1f} BPM",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv2.imshow("open-rppg camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()