import os
from PIL import Image

def validate_dataset(dataset_path="rice_dataset"):
    classes = os.listdir(dataset_path)
    print(f"Found {len(classes)} classes: {classes}")
    
    for cls in classes:
        images = os.listdir(f"{dataset_path}/{cls}")
        print(f"\nClass {cls}: {len(images)} images")
        
        # Verify first 3 images
        for img in images[:3]:
            try:
                Image.open(f"{dataset_path}/{cls}/{img}").verify()
            except Exception as e:
                print(f"Invalid image: {cls}/{img} - {str(e)}")

if __name__ == "__main__":
    validate_dataset()