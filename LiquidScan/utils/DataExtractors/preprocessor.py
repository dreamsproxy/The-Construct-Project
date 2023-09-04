import cv2
from glob import glob
from tqdm import tqdm

class cam:
    def __init__(self, cam_dir, verbose_level = 0) -> str:
        self.cam_dir = cam_dir
        self.cam_data = list()
        self.CAM_DATA_RESTRAINT = 12
        self.verbose_level = verbose_level
        return self
    
    def extract_data(self):
        files_list = glob(self.cam_dir)
        if self.verbose_level == 1:
            print("Reading files")
        for idx, f in tqdm(enumerate(files_list)):
            # Prepare holder
            structure_cache = []
            structure_cache.append(idx)
            structure_cache.append(f.replace("\\", "/"))

            # Load, read
            with open(f, "r") as infile:
                read_cache = infile.read().replace("\n", " ").split(" ")

            # prepare nested cache that will be append into
            #   `structure_cache`
            nested_cache = []
            
            # remove empty
            for value in read_cache:
                if len(value) >= 1:
                    nested_cache.append(value)
            # Reduce memory load
            del read_cache

            # remove 2D pos, focal length, camera ID, camera scales
            nested_cache = nested_cache[:-6]
            structure_cache.append(nested_cache)
            if len(nested_cache) > 12:
                print("The data you supplied has more than 12 values! Heres what the data should look like!")
                print("Translation: [X, Y, Z]")
                print("Rotation matrix 3x3:")
                print("[[var, var, var],")
                print("[var, var, var],")
                print("[var, var, var]]")
                raise Exception("There should only be 12 points of data\n\tafter removing the last 6 values!\n\tplease check your data")
            # Save memory
            del nested_cache
            self.cam_data.append(structure_cache)
            # Save memory
            del structure_cache
        
        return self.cam_data

    def verbose_return(self):
        print(f"Cam data directory: {self.cam_dir}")
        print(f"N-Variable restrains: {self.CAM_DATA_RESTRAINT}")
        print("")
        print("Data Below")
        print(self.cam_data)

    def example(self):
        sample = "./dataset/preprocessing/VALVE_DS/cam_pos/*.cam"
        cam_class = cam(sample)
        cam_class.extract_data()
        print(cam_class.cam_data[0][2])