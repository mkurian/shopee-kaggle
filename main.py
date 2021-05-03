import Explorer

if __name__ == '__main__':
    imageHandler = Explorer.ImageHandler()

    path = '/Users/merrin/Work/Kaggle/shopee-product-matching'
    dataLoader = Explorer.DataLoader(path)
    # dataLoader.generate_image_path()

    train_df = dataLoader.load_df('train_proc.csv', 'train_images')
    test_df = dataLoader.load_df('test_proc.csv', 'test_images')
    print(train_df.shape)
    # imageHandler.plot_image_pairs(train_df, 11, 12)
    # imageHandler.plot_image(train_df, 11)

    # Since the process of composing a matrix is quite resource-intensive,
    # for clarity, we will take only the first thousand images.
    train_1000 = train_df.iloc[:1000, :]
    train_1000 = train_1000.drop(columns=['Unnamed: 0'])
    matchFinder = Explorer.MatchFinder()
    matches = matchFinder.match_matrix(train_1000['image_phash'])

    generator = Explorer.GeneratePairs()
    generator.generate_matching_pairs(path, matches)