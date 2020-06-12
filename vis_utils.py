import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotFromVoxels(voxels):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    
    
def plotImg(img):
	plt.figure(figsize=(4, 4))
	plt.imshow(img)
	plt.show()
	
	
# function to plot a list of images
def plot_image_list(imgs, columns = 4, size = 4):
    
    img_count = len(imgs)
    
    assert img_count > 0
    
    rows = img_count // columns if img_count % columns == 0 else img_count // columns + 1
    
    fig=plt.figure(figsize=(size*columns, size*rows))
    
    for i in range(1, img_count + 1):
        fig.add_subplot(rows, columns, i)
        img = imgs[i-1]
        
        
        # BGR to RGB
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        elif img.shape[-1] == 4:
            img[:, :, 0:3] = cv2.cvtColor(img[:, :, 0:3], cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        elif len(img.shape) < 3:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
            
    
    plt.show()
