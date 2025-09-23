if __name__ == '__main__':
    try:
        import wikipediaapi
        print('wikipediaapi import OK:', wikipediaapi.__file__)
    except Exception as e:
        print('wikipediaapi import ERROR:', type(e).__name__, e)
