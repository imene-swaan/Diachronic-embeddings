import tomli


def main():
    with open("config.toml") as f:
        config = tomli.load(f)

    print(config["name"])
    print(config["age"])
    print(config["favorite_foods"])











