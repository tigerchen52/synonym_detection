def baike_invoke():
    import baike_crawler_model
    print(baike_crawler_model.baike_search(('凤梨', '001')))

if __name__ == '__main__':
    baike_invoke()