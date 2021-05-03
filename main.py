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
    train_1000 = train_df.iloc[:1000, :]
    train_1000 = train_1000.drop(columns=['Unnamed: 0'])
    matchFinder = Preprocessor.MatchFinder()
    matches = matchFinder.match_matrix(train_1000['image_phash'])

    generator = Preprocessor.GeneratePairs()
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

if __name__ == '__main__':
    path = '/Users/merrin/Work/Kaggle/shopee-product-matching'

    #start with 'merged.csv' in path



