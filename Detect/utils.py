import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from django.core.mail import EmailMessage
import json



Image_Size = 256
Batch_Size = 32
Channels = 3
Epochs = 50
dataset_path = "Utilities/Dataset/real_and_fake_face"

def predict(image):
    try:
        model = tf.keras.models.load_model('Utilities/Model/face_detection_model.keras')
        img = tf.keras.preprocessing.image.load_img(image, target_size=(Image_Size, Image_Size))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        output_layer = model.layers[-1]

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = round(100 * (np.max(predictions[0])), 2)

        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence}%")

        return {'prediction' : predicted_class, 'confidence' : confidence}
    except ValueError:  
        loadDataset()
    

def loadDataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        shuffle=True,
        image_size=(Image_Size, Image_Size),
        batch_size=Batch_Size
    )

    class_names = dataset.class_names

    for image_batch, label_batch in dataset.take(1):
        print(image_batch.shape)
        print(label_batch.numpy())
    
    trainModel(dataset)


def splitting_dataset_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    
    ds_size=len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
        
    train_size=int(train_split * ds_size)
    val_size= int(val_split * ds_size)
    
    train_ds= ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


def trainModel(dataset):
    train_ds, val_ds, test_ds=splitting_dataset_tf(dataset)
    print(len(train_ds),len(val_ds),len(test_ds))

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(Image_Size, Image_Size),
        layers.Rescaling(1.0/255)
    ])

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    input_shape = (Batch_Size,Image_Size, Image_Size,Channels)
    n_classes = 3

    model = models.Sequential([
    resize_and_rescale,
    data_aug,
    layers.Conv2D(32, (3,3), activation='relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation= 'softmax'),
    
    ])

    model.build(input_shape=input_shape)


    model.compile(
        optimizer='adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )


    history = model.fit(
        train_ds,
        epochs=10,
        batch_size=Batch_Size,
        verbose=1,
        validation_data=val_ds
    )


    scores = model.evaluate(test_ds)

    model.save("face_detection_model.keras")



def SendMail(data):
    
    with open('Detect\mail-template.JSON', 'r') as file:
        mail_data = json.load(file)
    
    mail_data['message'] = mail_data['message'].replace('{email}', data.user.email)
    mail_data['message'] = mail_data['message'].replace('{confidence}', str(data.confidence))  # Convert to string if needed
    mail_data['message'] = mail_data['message'].replace('{name}', data.user.first_name + " " + data.user.last_name)
    mail_data['message'] = mail_data['message'].replace('{phone}', data.user.phone)
    mail_data['subject'] = mail_data['subject'].replace('{name}', data.user.first_name + " " + data.user.last_name)
    subject = mail_data['subject']
    message = mail_data['message']
    from_email = 'admin@ImageDetection.com'
    recipient_list = [mail_data['mail-to']]

    email = EmailMessage(subject, message, from_email, recipient_list)

    file_path = 'media/'+str(data.image)
    email.attach_file(file_path)

    email.send(fail_silently=False)



























# # Use a specific image for prediction
# image_path = "real_and_fake_face/training_fake/mid_473_0011.jpg"
# img = tf.keras.preprocessing.image.load_img(image_path, target_size=(Image_Size, Image_Size))
# img_array = tf.keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)

# # Load the saved model
# model = tf.keras.models.load_model("face_detection_model0.keras")

# output_layer = model.layers[-1]

# # Predict the class and confidence
# predictions = model.predict(img_array)
# predicted_class = np.argmax(predictions[0])
# # predicted_class = class_names[np.argmax(predictions[0])]
# confidence = round(100 * (np.max(predictions[0])), 2)

# # Display the prediction results
# print(f"Predicted Class: {predicted_class}")
# print(f"Confidence: {confidence}%")


# print(predicted_class)
# # Display the image
