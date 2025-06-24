# face_matcher.py

import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

class FaceMatcher:
    def __init__(self, det_size=(640, 640), similarity_threshold=0.38):
        print("[🔧] Initializing Face Analysis model...")
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.similarity_threshold = similarity_threshold

    def capture_face(self):
        print("[📷] Opening webcam... Press SPACE to capture")
        cap = cv2.VideoCapture(0)
        captured_image = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            cv2.imshow("Press SPACE to capture", frame)
            key = cv2.waitKey(1)
            if key == 32:  # SPACE
                captured_image = frame.copy()
                print("[✅] Image captured!")
                break

        cap.release()
        cv2.destroyAllWindows()
        return captured_image

    def get_embedding(self, image):
        faces = self.app.get(image)
        if not faces:
            print("❌ No face detected.")
            return None
        return faces[0].embedding.reshape(1, -1)
    

    def find_similar_faces(self, query_embedding, database_path="./image_database"):
        matched_images = []
        results_path = "results"
        os.makedirs(results_path, exist_ok=True)

        print("\n🔍 Searching for matching faces in database...")

        # Clear previous results
        for file in os.listdir(results_path):
            os.remove(os.path.join(results_path, file))


        for filename in os.listdir(database_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(database_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = self.app.get(img)
                for face in faces:
                    if face.embedding is not None:
                        score = cosine_similarity(query_embedding, face.embedding.reshape(1, -1))[0][0]
                        if score >= self.similarity_threshold:
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, face.bbox)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Save the annotated image
                            result_img_path = os.path.join(results_path, filename)
                            cv2.imwrite(result_img_path, img)

                            matched_images.append(result_img_path)
                            print(f"✅ Match: {filename} | Similarity: {score:.3f}")
                            break

        return matched_images



    def show_matches(self, matched_images):
        if not matched_images:
            print("\n❌ No matching images found.")
            return

        print("\n🎯 Matching images:")
        for img_path, bbox in matched_images:
            img = cv2.imread(img_path)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(f"Match: {os.path.basename(img_path)}", img)

        print("\n[🛑] Press any key to close all windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    matcher = FaceMatcher()

    # Step 1: Capture Image
    captured_image = matcher.capture_face()
    if captured_image is None:
        exit()

    # Step 2: Get Embedding
    query_embedding = matcher.get_embedding(captured_image)
    if query_embedding is None:
        exit()

    # Step 3: Search in Database
    matches = matcher.find_similar_faces(query_embedding)

    # Step 4: Show results
    matcher.show_matches(matches)
