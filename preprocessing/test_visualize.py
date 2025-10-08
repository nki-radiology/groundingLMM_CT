# import matplotlib.pyplot as plt
# import numpy as np

# # Load a slice from the saved numpy array
# # slice_data = np.load("/home/r.vt.hull/preprocess_totalseg/data/combined_seg/s0003_300_combined.npy")  # Replace with your file path

# for i in range(250,260):
#     # slice_data = np.load(f"/home/r.vt.hull/preprocess_totalseg/data/combined_seg/s0010_{i}_combined.npy")
#     seg_data = np.load(f"/home/r.vt.hull/data/Semantic_Segm/totalseg/train/combined_seg/s0010_{i}_combined.npy")
#     ct_data = np.load(f"/home/r.vt.hull/data/Semantic_Segm/totalseg/train/ct/s0010_{i}.npy")
#     # Visualize the slice using Matplotlib
#     plt.imshow(seg_data, cmap='gray')  # Use 'gray' colormap for CT images
#     plt.imshow(ct_data, cmap='gray')  # Use 'gray' colormap for CT images
#     plt.axis('off')  # Turn off axis labels
#     plt.tight_layout(pad=0)
#     plt.savefig(f'test_images/combined_{i}.png',bbox_inches='tight', pad_inches=0)
#     plt.show()

# # slice_data = np.load("/home/r.vt.hull/preprocess_totalseg/data/seg/s1307_340_lungs.npy")  # Replace with your file path

# # # Visualize the slice using Matplotlib
# # plt.imshow(slice_data, cmap='gray')  # Use 'gray' colormap for CT images
# # plt.axis('off')  # Turn off axis labels
# # plt.title('CT Slice')  # Set the title
# # plt.savefig(f'test_lungs.png')
# # plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Load a slice from the saved numpy array
for i in range(250, 260):
    seg_data = np.load(f"/home/r.vt.hull/data/Semantic_Segm/totalseg/train/combined_seg/s0010_{i}_combined.npy")
    ct_data = np.load(f"/home/r.vt.hull/data/Semantic_Segm/totalseg/train/ct/s0010_{i}.npy")
    
    fig, axes = plt.subplots(1, 2)  # Create subplots with 1 row and 2 columns

    axes[0].imshow(ct_data, cmap='gray')  # Display CT data on the second subplot
    axes[0].axis('off')  # Turn off axis labels for the second subplot
    axes[0].set_title('CT')  # Set title for the second subplot


    axes[1].imshow(seg_data, cmap='gray')  # Display segmentation data on the first subplot
    axes[1].axis('off')  # Turn off axis labels for the first subplot
    axes[1].set_title('Segmentation')  # Set title for the first subplot

    # plt.tight_layout(pad=0)  # Adjust layout
    plt.savefig(f'test_images/combined_{i}.png', bbox_inches='tight')
    plt.show()
