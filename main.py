import Explorer
import os

if __name__ == '__main__':
    path = '/Users/merrin/Work/Kaggle/shopee-product-matching'

    print(os.listdir(path))

    # dataLoader = Explorer.DataLoader(path)
    # train_df = dataLoader.load_train_df(path, 'train.csv', 'train_images')
    #
    # print(train_df.loc[[11, 12], ['posting_id', 'image_phash', 'title', 'label_group']])
    # print(train_df.loc[[889, 890, 891], ['posting_id', 'image_phash', 'title', 'label_group']])
    #
    # imageHandler = Explorer.ImageHandler()
    # imageHandler.plot_image_pairs(train_df, 11, 12)
