from pathlib import Path

import psutil

from grcnn.processing import filter_bad_samples
from grcnn.training import keras, train
from grcnn.utils import find_best_model_epoch, find_model_path


def main():
    dataset_dir_path = '/srv/data_science/storage/product_code_ocr'
    max_text_len = 26
    w, h = 100, 32
    df_name = f'train_set_len{max_text_len}_filtered2'
    save_df_name = f'train_set_len{max_text_len}_filtered3'
    model_num = 47
    batch_size = 80
    image_dir = f'images_train_padding_size{w}x{h}'
    epoch = find_best_model_epoch(model_num)
    filter_bad_samples(dataset_dir_path, df_name, save_df_name, model_num, epoch, max_text_len, batch_size, image_dir)

    batch_size = 48
    model_num = 48
    initial_epoch = 0
    width, height = 100, 32
    max_text_len = 26
    n_classes = 10
    epochs = 200
    grcl_niter = 3
    grcl_fsize = 3
    lstm_units = 512
    train_augment_prob = .5
    lr = .0001
    optimizer = keras.optimizers.Adam(lr)
    # pretrain_model_num = 47
    # pretrain_model_path = Path(f'/srv/data_science/training/checkpoints/product_code_ocr/model{pretrain_model_num}')
    # pretrain_model_path = find_model_path(pretrain_model_path, find_best_model_path(pretrain_model_path))
    pretrain_model_path = None
    img_dir = f'images_train_padding_size{width}x{height}'
    train_df_name = f'train_set_len{max_text_len}_filtered3'
    val_df_name = f'val_set_len{max_text_len}_filtered'
    train(train_df_name, val_df_name, img_dir,
          model_num, initial_epoch, epochs, train_augment_prob,
          batch_size, pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units)

    batch_size = 48
    model_num = 49
    initial_epoch = 0
    width, height = 100, 32
    max_text_len = 26
    n_classes = 10
    epochs = 200
    grcl_niter = 3
    grcl_fsize = 3
    lstm_units = 512
    train_augment_prob = .5
    pretrain_model_num = 48
    lr = .1
    optimizer = keras.optimizers.Adadelta(lr, rho=.9)
    pretrain_model_path = Path(f'/srv/data_science/training/checkpoints/product_code_ocr/model{pretrain_model_num}')
    pretrain_model_path = find_model_path(pretrain_model_path, find_best_model_epoch(pretrain_model_num))
    img_dir = f'images_train_padding_size{width}x{height}'
    train_df_name = f'train_set_len{max_text_len}_filtered3'
    val_df_name = f'val_set_len{max_text_len}_filtered'
    train(train_df_name, val_df_name, img_dir,
          model_num, initial_epoch, epochs, train_augment_prob,
          batch_size, pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units)


def kill_process(process_id):
    try:
        p = psutil.Process(process_id)
    except psutil.NoSuchProcess:
        return
    p.kill()


if __name__ == '__main__':
    main()
