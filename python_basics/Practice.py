

def like_post():
    like_post.count +=1
    print(f"Total likes: {like_post.count} ")

def log_workout(calories):
    log_workout.count +=calories
    print(f"Total calories burned: {log_workout.count} calories")

def discount(product_price):
    dis = lambda product_price1: product_price * 0.9 if product_price > 100 else product_price
    print(f"Discounted price: {dis(product_price)}")
    return dis
def review_count(price, count_review):
    rev = lambda count_review1: price*0.80 if len(count_review)>100 else price*0.90
    print(f"Revised review count: {rev(count_review)}")

def weather_app(temp):
    celsius = list(map(lambda f: (f-32)*5/9, temp))
    print(celsius)

def steps_count(steps):

    more_steps = list(filter(lambda f:f>10000, steps))
    print(f"More steps: {more_steps}")

def add_revenue(revenue):
    from functools import reduce
    total = reduce(lambda t,y:t+y, revenue)
    print(f"Total revenue: {total}")

if __name__ == '__main__':
    tempr = [12, 34, 67, 56]
    add_revenue(tempr)