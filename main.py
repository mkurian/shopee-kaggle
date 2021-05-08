import Preprocessor
import pandas as pd


def initial_processing(path):
    dataLoader = Preprocessor.DataLoader(path)
    imageHandler = Preprocessor.ImageHandler()

    dataLoader.generate_image_path()

    train_df = dataLoader.load_df('train_proc.csv', 'train_images')
    test_df = dataLoader.load_df('test_proc.csv', 'test_images')
    print(train_df.shape)
    imageHandler.plot_image_pairs(train_df, 11, 12)
    imageHandler.plot_image(train_df, 11)

    # Since the process of composing a matrix is quite resource-intensive,
    # for clarity, we will take only the first thousand images.
    # TODO:
    # A good strategy to expand on this matching dataset
    # is to look for same labelgroup and generate matching pairs within each labelgroup

    train_1000 = train_df.iloc[:1000, :]
    train_1000 = train_1000.drop(columns=['Unnamed: 0'])
    matchFinder = Preprocessor.MatchFinder()
    matches = matchFinder.match_matrix(train_1000['image_phash'])

    generator = Preprocessor.GenerateDataset()
    generator.generate_matching_pairs(path, matches)

    # We have a dataset generated with first 1000 entries from the train dataset
    df1 = pd.read_csv(path + '/matching_pairs.csv')
    df1['label'] = True
    df2 = pd.read_csv(path + '/non_matching_pairs.csv')
    df1 = df1.drop(columns=['Unnamed: 0'])
    df2 = df2.drop(columns=['Unnamed: 0'])
    df2['label'] = False
    merged = pd.concat([df1, df2])
    merged = merged.reset_index(drop=True)
    merged.to_csv(path + '/merged.csv')

    # start with 'merged.csv' in path
    index_df = pd.read_csv(path + '/merged.csv')
    train_df = pd.read_csv(path + '/train_proc.csv')
    train_df = train_df.drop(columns=['Unnamed: 0'])
    index_df = index_df.drop(columns=['Unnamed: 0'])
    train_1000 = train_df.iloc[:1000, :]
    generator = Preprocessor.GenerateDataset()
    merged_df = generator.merged_dataset(index_df, train_1000)
    pd.set_option('display.max_columns', None)
    merged_df = merged_df.reset_index(drop=True)
    merged_df.to_csv(path + '/merged_with_data.csv')


if __name__ == '__main__':
    path = '/Users/merrin/Work/Kaggle/shopee-product-matching'
    merged = pd.read_csv(path + '/merged_with_data.csv')
    merged = merged.drop(columns=['Unnamed: 0'])

    imageHandler = Preprocessor.ImageHandler()
    merged['tensor_1'] = path + '/train/tensor/' + merged['path_1'].apply(lambda s: s.rsplit('/', 1)[1] + '.pt')
    merged['tensor_2'] = path + '/train/tensor/' + merged['path_2'].apply(lambda s: s.rsplit('/', 1)[1] + '.pt')

    merged.apply(lambda row: imageHandler.standardize_image(200, 200, row['path_1'], row['tensor_1']), axis=1)
    merged.apply(lambda row: imageHandler.standardize_image(200, 200, row['path_2'], row['tensor_2']), axis=1)
