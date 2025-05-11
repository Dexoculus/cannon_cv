import os
import shutil
import random

def split_dataset(source_dir, dest_train, dest_valid, dest_test,
                  train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train + valid + test musrt equal 1.0")

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        file_list = os.listdir(class_path)
        random.shuffle(file_list)

        n_total = len(file_list)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)

        train_files = file_list[:n_train]
        valid_files = file_list[n_train:n_train + n_valid]
        test_files = file_list[n_train + n_valid:]

        # 각 클래스 폴더 생성
        for dest, file_group in zip(
            [dest_train, dest_valid, dest_test],
            [train_files, valid_files, test_files]
        ):
            class_dest_dir = os.path.join(dest, class_name)
            os.makedirs(class_dest_dir, exist_ok=True)

            for file_name in file_group:
                src_path = os.path.join(class_path, file_name)
                dst_path = os.path.join(class_dest_dir, file_name)
                shutil.copyfile(src_path, dst_path)

    print("Dataset split into train/valid/test complete.")


split_dataset(
    source_dir="./dataset/augmented",
    dest_train="./dataset/train",
    dest_valid="./dataset/valid",
    dest_test="./dataset/test",
    train_ratio=0.75,
    valid_ratio=0.125,
    test_ratio=0.125
)
