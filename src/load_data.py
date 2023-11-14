from os import getenv
import dotenv
import re
import praw


dotenv.load_dotenv()


def carrega_dados(assuntos):
    
    api_reddit = praw.Reddit(client_id = getenv('client_id'),
                             client_secret = getenv('client_secret'),
                             password = getenv('password'),
                             user_agent = getenv('user_agent'),
                             username = getenv('user'))

    char_count = lambda post: len(re.sub('\W|\d', '', post.selftext))

    mask = lambda post: char_count(post) >= 100

    data = []
    labels = []

    for i, assunto in enumerate(assuntos):

        subreddit_data = api_reddit.subreddit(assunto).new(limit=1000)

        posts = [post.selftext for post in filter(mask, subreddit_data)]

        data.extend(posts)
        labels.extend([i] * len(posts))

        print(f'Número de posts do assunto r/{assunto}: {len(posts)}',
              f'\nUm dos posts extráidos: {posts[0][:600]}...\n',
              "_" * 80 + '\n')
    
    return data, labels