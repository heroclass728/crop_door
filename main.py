from utils.door_frame import crop_door_from_image
from utils.train_data_x_y import TrainData


if __name__ == '__main__':

    train_data = TrainData()
    crop_door_from_image()
    train_data.prepare_train_data()
