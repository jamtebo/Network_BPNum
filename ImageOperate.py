from PIL import Image
import imageio

def GetImageData(nWidth, nHeight, strPathIn):
    # 待处理图片存储路径
    im = Image.open(strPathIn)
    # Resize图片大小，入口参数为一个tuple，新的图片大小
    imReshape = im.resize((nWidth, nHeight))
    imReshape.save("Tmp/Tmp.png")
    img_array = imageio.imread("Tmp/Tmp.png", as_gray=True)
    img_data = 255.0 - img_array
    img_data = (img_data / 255.0 * 0.99) + 0.01
    return img_data
