import argparse
from candidate_generation import CandidateGeneration
from vehicle_model_matching import VehicleModelMatching
from parameter_finetune import  ParameterFinetune

parser = argparse.ArgumentParser(description='Camera calibration algorithm for the traffic monitoring camera')
parser.add_argument("--camera_code", type=str, default="K109F")
parser.add_argument("--image_height", type=int, default=240)
parser.add_argument("--image_width", type=int, default=320)
parser.add_argument("--focal_length", type=int, default=350)
parser.add_argument("--alpha", type=float, default=6)
parser.add_argument("--temperature", type=int, default=0)
parser.add_argument("--budget_1", type=int, default=2000)
parser.add_argument("--budget_2", type=int, default=20000)
parser.add_argument("--use_ransac", type=bool, default=False)
parser.add_argument("--opt_name", type=str, default="CMA")
parser.add_argument("--verbose", type=bool, default=False)

parser.add_argument("--object_points_path", type=str, default="../examples/camera_calibration/K109F/objects/vehicle_3dkeypoints_head.json")
parser.add_argument("--image_points_dir", type=str, default="../examples/camera_calibration/K109F/labels")
parser.add_argument("--images_dir", type=str, default="../examples/camera_calibration/K109F/images")
parser.add_argument("--save_dir", type=str, default="../examples/camera_calibration/K109F/outputs")

args = parser.parse_args()

config = {}
config["camera_code"] = args.camera_code
config["image_height"] = args.image_height
config["image_width"] = args.image_width
config["focal_length"] = args.focal_length
config["alpha"] = args.alpha
config["temperature"] = args.temperature
config["budget_1"] = args.budget_1
config["budget_2"] = args.budget_2
config["use_ransac"] = args.use_ransac
config["opt_name"] = args.opt_name

config["object_points_path"] = args.object_points_path
config["image_points_dir"] = args.image_points_dir
config["images_dir"] = args.images_dir
config["save_dir"] = args.save_dir

config["verbose"] = args.verbose

cg = CandidateGeneration(config)
vmm = VehicleModelMatching(config)
pf = ParameterFinetune(config)

cg.run()
vmm.run()
pf.run()