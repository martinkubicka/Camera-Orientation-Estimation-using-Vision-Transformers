import os
import cv2

PANO_SRC_BASE = "/Users/martinkubicka/Desktop/new_pano_dataset"

folders = [
    folder for folder in os.listdir(PANO_SRC_BASE)
    if os.path.isdir(os.path.join(PANO_SRC_BASE, folder))
]

print(f"Found {len(folders)} folders to inspect.\n")
print("Controls:")
print("  Q = Print folder name (without 'lsar_') and go to next")
print("  N = Skip to next")
print("  ESC = Exit early\n")

for folder in folders:
    snapshot_path = os.path.join(PANO_SRC_BASE, folder, "cyl", "snapshot.png")

    if not os.path.exists(snapshot_path):
        print("Snapshot not found:", folder.replace("lsar_", ""))
        continue

    img = cv2.imread(snapshot_path)
    if img is None:
        print("Cant read the image:", folder.replace("lsar_", ""))
        continue

    cv2.imshow("Snapshot Viewer", img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("DELETE:", folder.replace("lsar_", ""))
        continue
    
    if key == ord('d') or key == ord('D'):
        print("Should delete:", folder.replace("lsar_", ""))
        continue

    elif key == ord('n') or key == ord('N'):
        continue

    elif key == 27:
        break

cv2.destroyAllWindows()
