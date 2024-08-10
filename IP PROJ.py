from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import messagebox, simpledialog
import tkinter as tk
import cv2 as cv
import numpy as np

image_edited = None
image = None
image2 = None

def Histogram_Equalization():
    global image, image_edited
    if isinstance(image, np.ndarray):
        hist = np.zeros((256), dtype=int)
        pdf = np.zeros((256), dtype=float)
        cdf = np.zeros((256), dtype=float)
        s = np.zeros((256), dtype=float)
        hist_equal_lev = np.zeros((256), dtype=float)
        for k in range(256):
            hist[k] = np.count_nonzero(image == k)
            pdf[k] = hist[k] / (image.shape[0] * image.shape[1])
            if k >= 1: cdf[k] = cdf[k - 1] + pdf[k]
            else: cdf[0] = pdf[0]
            s[k] = cdf[k] * 255
            hist_equal_lev[k] = round(s[k])
        new_img = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                new_img[i, j] = hist_equal_lev[image[i, j]]
        image_edited = new_img
        cv.imshow("Histogram Equalization", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Nearest_neighbor_Interpolation():
    global image, image_edited
    if isinstance(image, np.ndarray):
        C = simpledialog.askinteger("C", "How many Columns?")
        R = simpledialog.askinteger("R", "How many Rows?")
        OrR, OrC = image.shape
        scale_row = R / OrR
        scale_col = C / OrC
        img2 = np.zeros((R, C), dtype=image.dtype)
        for r in range(R):
            for c in range(C):

                pos_r = int(r/ scale_row)
                pos_c = int(c / scale_col)

                img2[r, c] = image[pos_r, pos_c]
        image_edited = img2
        cv.imshow("Nearest Neighbor Interpolation Edited", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Salt_and_Paper():
    global image, image_edited
    if isinstance(image, np.ndarray):
        g = np.zeros_like(image)
        row, col = image.shape
        p = 0.09
        s = 1 - p
        for i in range(row):
            for j in range(col):
                rbn = np.random.uniform()
                if rbn < p:
                    g[i][j] = 0
                elif rbn > s:
                    g[i][j] = 255
                else:
                    g[i][j] = image[i][j]
        image_edited = g
        cv.imshow("Salt And Paper", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def DFT_And_IDFT():
    global image, image_edited
    if isinstance(image, np.ndarray):
        def DFT2D(image):
            M, N = image.shape
            F = np.zeros((M, N), dtype=complex)
            for u in range(M):
                for v in range(N):
                    sum = 0
                    for x in range(M):
                        for y in range(N):
                            sum += image[x, y] * np.exp(-2j * np.pi * (u * x / M + v * y / N))
                    F[u, v] = sum
            return F
        def IDFT2D(F):
            M, N = F.shape
            image = np.zeros((M, N), dtype=complex)
            for x in range(M):
                for y in range(N):
                    sum = 0
                    for u in range(M):
                        for v in range(N):
                            sum += F[u, v] * np.exp(2j * np.pi * (u * x / M + v * y / N))
                    image[x, y] = sum / (M * N)
            return image
        F = DFT2D(image)
        magnitude_spectrum = np.log(np.abs(F) + 1)
        reconstructed_image = IDFT2D(F)
        cv.imshow("DFT_And_IDFT", magnitude_spectrum)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imshow("DFT_And_IDFT", reconstructed_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Gaussian():
    global image, image_edited
    if isinstance(image, np.ndarray):
        x, y = image.shape
        mean = 5
        var = 200
        sigma = np.sqrt(var)
        n = np.random.normal(loc=mean, scale=sigma, size=(x,y))
        image_edited = image + n
        cv.imshow("Histogram Equalization", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Uniform():
    global image, image_edited
    if isinstance(image, np.ndarray):
        row, col = image.shape
        a = 0
        b = 50
        n = np.zeros((row, col), dtype=np.float64)
        for i in range(row):
            for j in range(col):
                n[i][j] = np.random.uniform(a, b)
        noise_img = image + n
        image_edited = noise_img
        cv.imshow("Noise image", n)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imshow("Noise + image", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def gaussian_smoothing_filter():
    global image, image_edited
    if isinstance(image, np.ndarray):
        kernel_size = simpledialog.askinteger("Kernel Size", "How big is kernel do you want?")
        sigma = simpledialog.askinteger("Sigma", "What value do you want sigma to be?")
        r, c = image.shape
        pad = np.zeros((r + kernel_size // 2, c + kernel_size // 2))
        pad[kernel_size // 2:r + kernel_size // 2, kernel_size // 2:c + kernel_size // 2] = image.copy()
        output_image = np.zeros_like(image)
        for i in range(kernel_size // 2, pad.shape[0] - kernel_size // 2):
            for j in range(kernel_size // 2, pad.shape[1] - kernel_size // 2):
                kernel = pad[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1]
                gaussian_kernel = np.zeros((kernel_size, kernel_size))
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        x = m - kernel_size // 2
                        y = n - kernel_size // 2
                        gaussian_kernel[m, n] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
                output_image[i - kernel_size // 2, j - kernel_size // 2] = np.sum(kernel * gaussian_kernel)
        image_edited = output_image
        cv.imshow("Gaussian Smoothing Filter", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def median_filter():
    global image, image_edited
    if isinstance(image, np.ndarray):
        r, c = image.shape
        kernel_size = simpledialog.askinteger("Kernel Size", "How big is kernel do you want?")
        kernlSize=kernel_size
        pad = np.zeros((r + kernel_size // 2, c + kernel_size // 2))
        pad[kernel_size // 2:r + kernel_size // 2, kernel_size // 2:c + kernel_size // 2] = image.copy()
        output_image = np.zeros_like(image)
        for rr in range(kernlSize//2,pad.shape[0]- kernlSize//2):
            for cc in range(kernlSize//2,pad.shape[1]- kernlSize//2):
                kernel=pad[rr-kernlSize//2 : rr+kernlSize//2 *2, cc-kernlSize//2 : cc+kernlSize//2*2]
                ar =kernel.flatten()
                ar.sort()
                l=len(ar)
                med=ar[l//2]
                output_image[rr-kernlSize//2,cc-kernlSize//2]=med
        image_edited = output_image
        cv.imshow("Median Filter", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def adaptive_filter():
    global image, image_edited
    if isinstance(image, np.ndarray):
        kernel_size = simpledialog.askinteger("Kernel Size", "How big is kernel do you want?")
        f = simpledialog.askinteger("f", "Choose 1 or 2 or 3")
        r, c = image.shape
        kernlSize=kernel_size
        pad = np.zeros((r + kernel_size // 2, c + kernel_size // 2))
        pad[kernel_size // 2:r + kernel_size // 2, kernel_size // 2:c + kernel_size // 2] = image.copy()
        output_image = np.zeros_like(image)
        if f == 3:
            for rr in range(kernlSize//2,pad.shape[0]- kernlSize//2):
                for cc in range(kernlSize//2,pad.shape[1]- kernlSize//2):
                    kernel=pad[rr-kernlSize//2 : rr+kernlSize//2 *2, cc-kernlSize//2 : cc+kernlSize//2*2]
                    ar =kernel.flatten()
                    ar.sort()
                    mix=ar[len(ar)-1]
                    output_image[rr-kernlSize//2,cc-kernlSize//2]=mix
        if f == 2:
            for rr in range(kernlSize//2,pad.shape[0]- kernlSize//2):
                for cc in range(kernlSize//2,pad.shape[1]- kernlSize//2):
                    kernel=pad[rr-kernlSize//2 : rr+kernlSize//2 *2, cc-kernlSize//2 : cc+kernlSize//2*2]
                    ar =kernel.flatten()
                    ar.sort()
                    min=ar[0]
                    output_image[rr-kernlSize//2,cc-kernlSize//2]=min
        if f == 1:
            for rr in range(kernlSize//2,pad.shape[0]- kernlSize//2):
                for cc in range(kernlSize//2,pad.shape[1]- kernlSize//2):
                    kernel=pad[rr-kernlSize//2 : rr+kernlSize//2 *2, cc-kernlSize//2 : cc+kernlSize//2*2]
                    ar =kernel.flatten()
                    ar.sort()
                    l=len(ar)
                    med=ar[l//2]
                    output_image[rr-kernlSize//2,cc-kernlSize//2]=med
        image_edited = output_image
        cv.imshow("Adaptive Filter", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def AVG_filter():
    global image, image_edited
    if isinstance(image, np.ndarray):
        r, c = image.shape
        kernel_size = simpledialog.askinteger("Kernel Size", "How big is kernel do you want?")
        kernlSize=kernel_size
        pad = np.zeros((r + kernel_size // 2, c + kernel_size // 2))
        pad[kernel_size // 2:r + kernel_size // 2, kernel_size // 2:c + kernel_size // 2] = image.copy()
        output_image = np.zeros_like(image)
        for rr in range(kernlSize//2,pad.shape[0]- kernlSize//2):
            for cc in range(kernlSize//2,pad.shape[1]- kernlSize//2):
                kernel=pad[rr-kernlSize//2 : rr+kernlSize//2 *2, cc-kernlSize//2 : cc+kernlSize//2*2]
                ar =kernel.flatten()
                sum=0
                for i in range(0,len(ar)):
                    sum+=ar[i]
                AVG=sum/len(ar)
                output_image[rr-kernlSize//2,cc-kernlSize//2]=AVG
        image_edited = output_image
        cv.imshow("AVG Filter", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Histogram_Specification():
    def Histogram_Equalization(img):
        if isinstance(img, np.ndarray):
            hist = np.zeros((256), dtype=int)
            pdf = np.zeros((256), dtype=float)
            cdf = np.zeros((256), dtype=float)
            s = np.zeros((256), dtype=float)
            hist_equal_lev = np.zeros((256), dtype=float)
            for k in range(256):
                hist[k] = np.count_nonzero(img == k)
                pdf[k] = hist[k] / (img.shape[0] * img.shape[1])
                if k >= 1: cdf[k] = cdf[k - 1] + pdf[k]
                else: cdf[0] = pdf[0]
                s[k] = cdf[k] * 255
                hist_equal_lev[k] = round(s[k])
            new_img = np.zeros_like(img)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    new_img[i, j] = hist_equal_lev[img[i, j]]
            return new_img, hist_equal_lev
        else:
            messagebox.showwarning("Warning", "Image missing")
    global image, image2, image_edited
    if isinstance(image, np.ndarray) and isinstance(image2, np.ndarray):
        src_img = Histogram_Equalization(image2)[0]
        ref_hist = Histogram_Equalization(image)[1]
        src_to_ref_mapping = np.zeros((256), dtype=int)
        for i in range(256):
            src_to_ref_mapping[i] = np.argmin(np.abs(i - ref_hist))
            matched_img = np.zeros_like(src_img)
            for i in range(src_img.shape[0]):
                for j in range(src_img.shape[1]):
                    matched_img[i, j] = src_to_ref_mapping[src_img[i, j]]
        cv.imshow("Histogram Specification", matched_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Laplacian_Operator_In_Spatial_Domain():
    global image, image_edited
    if isinstance(image, np.ndarray):
        kernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
        filtered_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filter_total = 0
                for k in range(kernel.shape[0]):
                    for l in range(kernel.shape[1]):
                        if 0 <= i + k < image.shape[0] and 0 <= j + l < image.shape[1]:
                            a = image[i + k, j + l]
                            b = kernel[k, l]
                            product = a * b
                            filter_total += product
                filtered_image[i, j] = filter_total
        c = -1
        enhanced_image = image + c * filtered_image
        gClip = np.clip(enhanced_image, 0, 255)
        image_edited = gClip
        cv.imshow("Laplacian Operator In Spatial Domain", image_edited)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Unsharp_Masking_and_Highboost_Filtering_In_Spatial_Domain():
    global image
    if isinstance(image, np.ndarray):
        img_normalized = image / 255.0
        img_blur = cv.GaussianBlur(src=img_normalized, ksize=(31, 31), sigmaX=0, sigmaY=0)
        g_mask = np.zeros_like(img_normalized)
        for i in range(img_normalized.shape[0]):
            for j in range(img_normalized.shape[1]):
                g_mask[i, j] = img_normalized[i, j] - img_blur[i, j]
        #highboost
        k = 5  
        g = np.zeros_like(img_normalized)
        for i in range(img_normalized.shape[0]):
            for j in range(img_normalized.shape[1]):
                g[i, j] = img_normalized[i, j] + k * g_mask[i, j]
                g[i, j] = np.clip(g[i, j], 0, 1)  
        cv.imshow("Unsharp Masking and Highboost Filtering In Spatial Domain", g)
        cv.waitKey(0)
        cv.destroyAllWindows()
        img_normalized = image / 255.0
        img_blur = cv.GaussianBlur(src=img_normalized, ksize=(31, 31), sigmaX=0, sigmaY=0)
        g_mask = np.zeros_like(img_normalized)
        for i in range(img_normalized.shape[0]):
            for j in range(img_normalized.shape[1]):
                g_mask[i, j] = img_normalized[i, j] - img_blur[i, j]
        #unsharp
        k = 1  
        g = np.zeros_like(img_normalized)
        for i in range(img_normalized.shape[0]):
            for j in range(img_normalized.shape[1]):
                g[i, j] = img_normalized[i, j] + k * g_mask[i, j]
                g[i, j] = np.clip(g[i, j], 0, 1)
        cv.imshow("Unsharp Masking and Highboost Filtering In Spatial Domain", g)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Roberts_Cross_Gradient_Operators_In_Spatial_Domain():
    global image, image_edited
    if isinstance(image, np.ndarray):
        img_normalized = image.astype('float64') / 255.0
        roberts_cross_v = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        roberts_cross_h = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        gradient_magnitude = np.zeros_like(img_normalized)
        for i in range(1, img_normalized.shape[0] - 1):
            for j in range(1, img_normalized.shape[1] - 1):
                gx = np.sum(img_normalized[i-1:i+2, j-1:j+2] * roberts_cross_h)
                gy = np.sum(img_normalized[i-1:i+2, j-1:j+2] * roberts_cross_v)
                gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)
        cv.imshow("Roberts Cross Gradient Operators In Spatial Domain", gradient_magnitude)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Sobel_Operator_In_Spatial_Domain():
    global image, image_edited
    if isinstance(image, np.ndarray):
        kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        output = np.zeros_like(image)
        height, width = image.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                gx = np.sum(image[y-1:y+2, x-1:x+2] * kernelx)
                gy = np.sum(image[y-1:y+2, x-1:x+2] * kernely)
                output[y, x] = np.sqrt(gx**2 + gy**2)
        cv.imshow("Sobel Operator In Spatial Domain", output)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showwarning("Warning", "Image missing")

def Huffman_Coding():
    global image
    if isinstance(image, np.ndarray):
        class Node:
            def __init__(self, right=None, left=None, parent=None, bit_str=None, freq=0) -> None:
                self.right = right
                self.left = left
                self.parent = parent
                self.bit_str = bit_str
                self.freq = freq
        def make_nodes(counted_pixels: dict[int, int]) -> list[Node]:
            nodes = []
            for i in range(len(counted_pixels)):
                node = Node(freq=counted_pixels[i][1], bit_str=str(counted_pixels[i][0]))
                nodes.append(node)
            return nodes
        
        def make_tree(nodes: list[Node]) -> Node:
            while len(nodes) != 1:
                left_node, right_node = nodes[0], nodes[1]
                new_node = Node(freq=left_node.freq + right_node.freq, left=left_node, right=right_node)
                left_node.parent, right_node.parent = new_node, new_node
                nodes.remove(left_node)
                nodes.remove(right_node)
                nodes.append(new_node)
                nodes = sorted(nodes, key=lambda node:node.freq)
            return nodes[0]

        def count_pixels(image: np.ndarray) -> dict[int, int]:
            row, col = image.shape
            num_pixels = {}
            for r in range(row):
                for c in range(col):
                    if image[r, c] not in num_pixels.keys():
                        num_pixels[image[r, c]] = 1
                    else:
                        num_pixels[image[r, c]] += 1
            return num_pixels

        def get_huffman_codes(node: Node, code=''):
            if node is None:
                return {}
            if node.left is None and node.right is None:
                return {node.bit_str: code}
            codes = {}
            codes.update(get_huffman_codes(node.left, code + '0'))
            codes.update(get_huffman_codes(node.right, code + '1'))
            return codes

        def compress_image(image: np.ndarray, codes: dict[str, str]) -> str:
            compressed_image = ""
            height, width = image.shape
            for i in range(height):
                for j in range(width):
                    pixel = str(image[i, j])
                    compressed_image += codes[pixel]
            return compressed_image

        def Huffman_coding(image: np.ndarray):
            counted_pixels = count_pixels(image)
            counted_pixels = sorted(counted_pixels.items(), key=lambda x:x[1])
            
            nodes: list[Node] = sorted(make_nodes(counted_pixels), key=lambda node:node.freq)
            head = make_tree(nodes)
            codes = get_huffman_codes(head)
            return compress_image(image, codes)

        compressed_image = Huffman_coding(image)
        with open("compress_image.txt", 'w') as f:
            f.write(compressed_image)

        messagebox.showinfo("Done", "The image has compressed")
    else:
        messagebox.showwarning("Warning", "Image missing")

root = tk.Tk()
SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
root.config(bg='#191919')
root.title("Image Processing Project")
root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
root.attributes('-fullscreen')

def select_image():
    """Makes the user select the desired image and returns it"""
    path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.bmp")])
    if path:
        global image
        image = cv.imread(path, 0)
        pil_image = Image.fromarray(image).resize((300, 300))
        image_tk = ImageTk.PhotoImage(image=pil_image)
        image_label.configure(image=image_tk)
        image_label.image = image_tk

def select_image2():
    """Makes the user select the desired image and returns it"""
    path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.bmp")])
    if path:
        global image2
        image2 = cv.imread(path, 0)
        pil_image = Image.fromarray(image2).resize((300, 300))
        image_tk = ImageTk.PhotoImage(image=pil_image)
        image_label2.configure(image=image_tk)
        image_label2.image = image_tk

select_image_button = tk.Button(master=root, text="Select Image", command=select_image)
select_image_button.place(x=400, y=400)

image_label = tk.Label(root, text="Please input an image")
image_label.place(x=270, y=50)

select_image_button2 = tk.Button(master=root, text="Select Image", command=select_image2)
select_image_button2.place(x=1000, y=400)

image_label2 = tk.Label(root, text="Please input another image if needed")
image_label2.place(x=850, y=50)

histo_button = tk.Button(master=root, text="Histogram Equalization", command=Histogram_Equalization)
histo_button.place(x=50, y=500)

nni = tk.Button(master=root, text="Nearest Neighbor Interpolation", command=Nearest_neighbor_Interpolation)
nni.place(x=220, y=500)

sap = tk.Button(master=root, text="Salt And Paper", command=Salt_and_Paper)
sap.place(x=430, y=500)

uni = tk.Button(master=root, text="Uniform", command=Uniform)
uni.place(x=550, y=500)

gsf = tk.Button(master=root, text="Gaussian Smoothing Filter", command=gaussian_smoothing_filter)
gsf.place(x=640, y=500)

medin = tk.Button(master=root, text="Median Filter", command=median_filter)
medin.place(x=830, y=500)

adapt = tk.Button(master=root, text="Adaptive Filter", command=adaptive_filter)
adapt.place(x=950, y=500)

avg = tk.Button(master=root, text="AVG Filter", command=AVG_filter)
avg.place(x=1050, y=500)

loisd = tk.Button(master=root, text="Laplacian Operator In Spatial Domain", command=Laplacian_Operator_In_Spatial_Domain)
loisd.place(x=1150, y=500)

umahfid = tk.Button(master=root, text="Unsharp Masking and Highboost Filtering In Spatial Domain", command=Unsharp_Masking_and_Highboost_Filtering_In_Spatial_Domain)
umahfid.place(x=300, y=600)

rcgoisd = tk.Button(master=root, text="Roberts Cross Gradient Operators In Spatial Domain", command=Roberts_Cross_Gradient_Operators_In_Spatial_Domain)
rcgoisd.place(x=650, y=600)

histospec = tk.Button(master=root, text="Histogram Specification", command=Histogram_Specification)
histospec.place(x=250, y=700)

soisd = tk.Button(master=root, text="Sobel Operator In Spatial Domain", command=Sobel_Operator_In_Spatial_Domain)
soisd.place(x=450, y=700)

dftidft = tk.Button(master=root, text="DFT And IDFT", command=DFT_And_IDFT)
dftidft.place(x=700, y=700)

gauss = tk.Button(master=root, text="Gaussian", command=Gaussian)
gauss.place(x=850, y=700)

huffman = tk.Button(master=root, text="Huffman Coding", command=Huffman_Coding)
huffman.place(x=950, y=700)

root.mainloop()