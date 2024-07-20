model = None
import os
import cv2
import numpy as np
import tensorflow as tf


def model_init():
    import tensorflow.keras as keras
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator,
        load_img,
        img_to_array,
    )
    from tensorflow.keras.models import Sequential
    from keras import utils
    from tensorflow.keras.layers import (
        Dense,
        Activation,
        Flatten,
        Dropout,
        BatchNormalization,
        Conv2D,
        MaxPooling2D,
    )
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras import regularizers, optimizers
    from tensorflow.keras.applications import (
        ResNet50,
        VGG16,
        ResNet101,
        VGG19,
        DenseNet201,
        EfficientNetB4,
        MobileNetV2,
    )

    from tensorflow.keras.applications import (
        resnet,
        vgg16,
        vgg19,
        densenet,
        efficientnet,
        mobilenet_v2,
    )
    from tensorflow.keras import Model
    from tensorflow.keras.optimizers.legacy import Adam
    import matplotlib.pyplot as plt
    import pandas as pd
    import PIL

    path = "/Users/iniyan/workspace/cancer_identification/Data/train"
    for files in os.listdir(path):
        print(os.path.join(path, files))

    train_path = "/Users/iniyan/workspace/cancer_identification/Data/train"
    valid_path = "/Users/iniyan/workspace/cancer_identification/Data/valid"
    test_path = "/Users/iniyan/workspace/cancer_identification/Data/test"

    image_paths = [
        "/Users/iniyan/workspace/cancer_identification/Data/train/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
        "/Users/iniyan/workspace/cancer_identification/Data/train/normal",
        "/Users/iniyan/workspace/cancer_identification/Data/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
        "/Users/iniyan/workspace/cancer_identification/Data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
    ]

    def load_images(image_paths=image_paths, n=36):
        # Load the images from disk.
        images = []
        for i in range(len(image_paths)):
            images_ = [
                cv2.imread(image_paths[i] + "/" + path)
                for path in os.listdir(image_paths[i])[: int(n / 4)]
            ]
            images.append(images_)
        # Convert to a numpy array and return it.
        sample = np.asarray(images, dtype=object)
        return sample

    sample = load_images()
    fig = plt.figure(figsize=(20, 5))
    l = 1
    shapes = []
    for i in range(sample.shape[0]):
        for m in range(sample.shape[1]):
            ax = fig.add_subplot(4, 9, m + l, xticks=[], yticks=[])
            ax.imshow(np.squeeze(sample[i, m]))
            shapes.append(sample[i, m].shape)
        l += 9

    np.array(shapes).mean(axis=0)

    image_shape = (305, 430, 3)
    N_CLASSES = 4
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(dtype="float32", rescale=1.0 / 255.0)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        batch_size=BATCH_SIZE,
        target_size=(305, 430),
        class_mode="categorical",
    )

    valid_datagen = ImageDataGenerator(dtype="float32", rescale=1.0 / 255.0)
    valid_generator = valid_datagen.flow_from_directory(
        valid_path,
        batch_size=BATCH_SIZE,
        target_size=(305, 430),
        class_mode="categorical",
    )

    test_datagen = ImageDataGenerator(dtype="float32", rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        batch_size=BATCH_SIZE,
        target_size=(305, 430),
        class_mode="categorical",
    )

    base_hidden_units = 8
    weight_decay = 1e-3
    model = Sequential(
        [
            Conv2D(
                filters=8,
                kernel_size=2,
                padding="same",
                activation="relu",
                input_shape=(image_shape),
            ),
            MaxPooling2D(pool_size=2),
            Conv2D(
                filters=16,
                kernel_size=2,
                padding="same",
                activation="relu",
                                input_shape=(image_shape),

                kernel_regularizer=regularizers.l2(weight_decay),
            ),
            MaxPooling2D(pool_size=2),
            Dropout(0.4),
            Flatten(),
            Dense(300, activation="relu"),
            Dropout(0.5),
            Dense(4, activation="softmax"),
        ]
    )

    model.summary()

    checkpointer = ModelCheckpoint("chestmodel.keras", verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=15)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    history = model.fit(
        train_generator,
        # steps_per_epoch=20,
        # epochs=100,
        steps_per_epoch=1,
        epochs=1,
        verbose=1,
        validation_data=valid_generator,
        callbacks=[checkpointer, early_stopping],
    )

    result = model.evaluate(test_generator)

    plt.plot(
        history.history["acc"],
        label="train",
    )
    plt.plot(history.history["val_acc"], label="val")

    plt.legend(loc="right")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()
    model.save("cancer_detection_model.keras")
