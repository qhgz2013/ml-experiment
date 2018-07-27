import requests
import base64


def main():
    print("Input an empty path to exit this program.")
    while True:
        path = input('Input image path: ')
        with open(path, 'rb') as file:
            length = file.seek(0, 2)
            file.seek(0)
            img_binary = file.read(length)
            img_binary = base64.encodebytes(img_binary)
            img = str(img_binary, 'utf-8')
        req = requests.post('http://localhost:10087/classify', data={'image': img})
        resp = str(req.content, 'utf-8')
        print(resp)


if __name__ == '__main__':
    main()
