import onnx
import onnxruntime
import numpy as np
import os
import telebot
import cv2

# Initialize your bot token
#TOKEN = '6251722523:AAEI7a-GW4dRneTM8LSnRgr-1swvpaAOAF0'
TOKEN = "6838811403:AAGUJFUwqR52m0O7u77fcH0L5Z0pk1ICdrM"
bot = telebot.TeleBot(TOKEN)

exec_path = os.getcwd()
#exec_path="drive/MyDrive/files/"
model_path = os.path.join(exec_path, "thebest.onnx")
model = onnx.load(model_path)
onnx.checker.check_model(model)

opt_session = onnxruntime.SessionOptions()

opt_session.enable_mem_pattern = True

opt_session.enable_cpu_mem_arena = True

opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

#EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
EP_list = ['AzureExecutionProvider','CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)

def image_preprocessing(input_shape,image):
    height, width = input_shape[2:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(width,height),interpolation = cv2.INTER_AREA)
    image = image / 255.0
    image = image.transpose(2,0,1)
    input_tensor = image[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor,height,width

def compute_iou(box, boxes):

    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(2, xmax - xmin) * np.maximum(2, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / union_area

    return iou


def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:

        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


@bot.message_handler(content_types=['text'])
def handle_message(message):

    mess = f"Salom 10Mb gacha bo'lgan videoni jo'nating"
    bot.send_message(message.chat.id, mess)

# Define the folder where videos will be saved
VIDEO_FOLDER = 'videos'

@bot.message_handler(content_types=['video'])
def handle_video(message):
    try:
        video_file = message.video
        print(video_file.file_size)
        if video_file.file_size > 10 * 1024 * 1024:  # 10MB in bytes
            bot.reply_to(message, "It's too big! Please send a smaller video.(10MB)")
        else:
            # Process the video or perform any other desired action
            bot.reply_to(message, "Video received. Thank you!")

        model_inputs = ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape

        model_output = ort_session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]
        CLASSES=["Face"]
        conf_thresold = 0.6

        # Create the video folder if it doesn't exist
        if not os.path.exists(VIDEO_FOLDER):
            os.makedirs(VIDEO_FOLDER)

        # Get the file info
        file_info = bot.get_file(message.video.file_id)
        file_path = file_info.file_path
        
        # Download the video
        downloaded_file = bot.download_file(file_path)
        msg=bot.reply_to(message, "video downloaded successfully")
        # Save the video to the folder
        video_filename = os.path.join(VIDEO_FOLDER, f"{message.video.file_id}.avi")
        output_file = os.path.join(VIDEO_FOLDER, f"out{message.video.file_id}.avi")
        
        with open(video_filename, 'wb') as new_file:
            new_file.write(downloaded_file)

        
        cap = cv2.VideoCapture(video_filename)
        print(video_filename)



        imageWidth = int(cap.get(3))
        imageHeight = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        freyms = int(cap.get(7))
        #print("fourcc=",cap.get(cv2.CAP_PROP_BITRATE))
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        out = cv2.VideoWriter(output_file, fourcc, fps, (imageWidth, imageHeight))
        k=0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            if k%2==0:
              k+=1
              continue
            k+=1
            try:
                if int(k/freyms*100)%10==0:
                    bot.edit_message_text(f"{int(k/freyms*100)} % completed...", msg.chat.id, msg.message_id)
            except :
                pass
            #frame=cv2.resize(frame, (640, 480),interpolation = cv2.INTER_AREA)
            input_tensor, h, w = image_preprocessing(input_shape,frame)
            output = ort_session.run(output_names, {input_names[0]: input_tensor})[0]

            predictions = np.squeeze(output).T
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > conf_thresold, :]
            scores = scores[scores > conf_thresold]

            class_ids = np.argmax(predictions[:, 4:], axis=1)

            boxes = predictions[:, :4]
            input_shape = np.array([w, h, w, h])
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array([imageWidth, imageHeight, imageWidth, imageHeight])
            boxes = boxes.astype(np.int32)
            indices = nms(boxes, scores, 0.3)

            image_draw = frame.copy()

            for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):

                bbox = bbox.round().astype(np.int32).tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                color = (0,255,0)
                cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            out.write(image_draw)

        bot.edit_message_text(f"the process is completed", msg.chat.id, msg.message_id)
        cap.release()
        out.release()
        # Send a confirmation message
        #res = bot.reply_to(message, "Loading result...")

        # Send the video back to the user
        try:
            with open(output_file, 'rb') as video_file:
                bot.send_video(message.chat.id, video_file)
        except Exception as er:            
            bot.reply_to(message, f"{str(er)}")
        # Delete the video file
        os.remove(video_filename)
        
        os.remove(output_file)


    except Exception as e:
        bot.reply_to(message, f"The file size is a large, please send a smaller one: {str(e)}")
    

if __name__ == "__main__":
    bot.polling()
