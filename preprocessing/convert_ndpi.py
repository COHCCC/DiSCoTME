import openslide
import tifffile as tiff
import numpy as np

def convert_ndpi_to_lossless_tiff(ndpi_path, output_tiff_path, tile_size=10240, compression="lzw"):
    """
    逐块读取 NDPI Level 0，并保存为无损压缩的 TIFF (LZW/ZLIB)
    
    Parameters
    ----------
    ndpi_path : str
        输入的 NDPI 文件路径
    output_tiff_path : str
        输出的 TIFF 文件路径
    tile_size : int
        每个 tile 读取大小 (默认 10240)
    compression : str
        TIFF 压缩方式 ("lzw" 或 "zlib")，推荐 LZW（兼容性好）
    """
    # 打开 NDPI 文件
    slide = openslide.OpenSlide(ndpi_path)

    # 获取 Level 0 (最高分辨率) 尺寸
    level = 0
    width, height = slide.level_dimensions[level]
    print(f"Extracting Level {level}: {width} x {height}")

    # 初始化大图 NumPy 数组
    image_array = np.zeros((height, width, 3), dtype=np.uint8)

    # **逐块读取并填充**
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)

            print(f"Reading tile at ({x}, {y}) size ({w}, {h})")
            tile = slide.read_region((x, y), level, (w, h)).convert("RGB")
            tile_np = np.array(tile, dtype=np.uint8)

            # 存入大 NumPy 数组
            image_array[y:y+h, x:x+w, :] = tile_np

    # **保存为 BigTIFF（无损压缩）**
    print("Saving as compressed TIFF...")
    tiff.imwrite(
        output_tiff_path,
        image_array,
        bigtiff=True,
        compression=compression  # "lzw" 或 "zlib"
    )
    print(f"Saved TIFF with {compression} compression: {output_tiff_path}")

if __name__ == "__main__":
    # **输入 NDPI 路径**
    ndpi_path = "/coh_labs/dits/nsong/Craig_VisiumHD_20241218/visium_ndpi/267_SP_21_232_C2-003.ndpi"
    
    # **输出 TIFF 路径**
    output_tiff_path = "/coh_labs/dits/nsong/Craig_VisiumHD_20241218/visium_ndpi/267_SP_21_232_C2-003_lzw.tiff"
    
    # **选择无损压缩方式 ("lzw" 推荐，"zlib" 也可)**
    compression_type = "lzw"  # 或 "zlib"

    # **转换 NDPI -> 无损 TIFF**
    convert_ndpi_to_lossless_tiff(ndpi_path, output_tiff_path, tile_size=10240, compression=compression_type)