import os
import json
import numpy as np
import nevergrad as ng
from concurrent import futures
from itertools import combinations
from camera_calibration.util_files.process import normalize, softmax
from camera_calibration.util_files.io import load_image_points, load_object_points

from base_model import BaseModel

class ParameterFinetune(BaseModel):
    def __init__(self, config):
        super(ParameterFinetune, self).__init__(config)
        self.object_points_full = None
        self.image_points_full = None
        self.fixed_object_points = None

    def load_vmm_results(self):
        with open(os.path.join(self.save_dir, "vmm_results.json")) as f:
            vmm_results = json.load(f)
        return vmm_results

    def get_back_projected_loss(self, object_points, back_projected_points, rvec, tvec, f):
        pairs = combinations(range(len(object_points)), 2)
        distance_loss = 0
        angle_loss = 0
        for each in pairs:
            object_vec = object_points[each[0]] - object_points[each[1]]
            back_projected_vec = back_projected_points[each[0]] - back_projected_points[each[1]]
            length_object = np.linalg.norm(object_vec)
            length_back_projected = np.linalg.norm(back_projected_vec)
            cos_theta = np.dot(object_vec, back_projected_vec.T) / (length_object * length_back_projected)

            distance_loss += np.square(length_object - length_back_projected)
            angle_loss += np.sqrt(1 - min(np.square(cos_theta), 1))
        pair_num = len(object_points) * (len(object_points) - 1) / 2
        return (distance_loss + self.loss_alpha * angle_loss) / pair_num, distance_loss, angle_loss

    def get_finetune_loss(self, fixed_object_points, rvec, tvec, f):
        losses = []
        back_projected_centriods = []
        for i, (object_points, image_points) in enumerate(zip(self.object_points_full, self.image_points_full)):
            back_projected_points = []
            for object_point, image_point in zip(object_points, image_points):
                X_hat, Y_hat = self.get_back_projected_point(object_point, rvec, tvec, f, object_point[-1])
                back_projected_points.append([X_hat, Y_hat, object_point[-1]])
            back_projected_points = np.array(back_projected_points)

            losses.append(self.get_back_projected_loss(object_points, back_projected_points,
                                                       rvec, tvec, f))
            back_projected_centriods.append(np.mean(back_projected_points, axis=0))

        fix_centriod = np.reshape(np.mean(fixed_object_points, axis=0), [1, 3])
        back_projected_centriods = np.array(back_projected_centriods)

        weights = np.linalg.norm(fix_centriod - back_projected_centriods, axis=1)
        weights = softmax(normalize(weights), self.loss_t)
        losses = np.array(losses)

        return np.sum(losses[:, 0] * weights), np.mean(losses[:, 0]), np.mean(losses[:, 1]), np.mean(losses[:, 2])

    def get_finetune_loss_cma(self, theta1, theta2, theta3, t1, t2, t3, f):
        rvec = np.array([[theta1], [theta2], [theta3]])
        tvec = np.array([[t1], [t2], [t3]])
        return self.get_finetune_loss(self.fixed_object_points, rvec, tvec, f)[0]

    def optimize(self, rvec, tvec, f):
        theta1 = ng.p.Scalar(init=rvec[0][0],
                             lower=rvec[0][0] - 0.3 * np.abs(rvec[0][0]),
                             upper=rvec[0][0] + 0.3 * np.abs(rvec[0][0]))
        theta2 = ng.p.Scalar(init=rvec[1][0],
                             lower=rvec[1][0] - 0.3 * np.abs(rvec[1][0]),
                             upper=rvec[1][0] + 0.3 * np.abs(rvec[1][0]))
        theta3 = ng.p.Scalar(init=rvec[2][0],
                             lower=rvec[2][0] - 0.3 * np.abs(rvec[2][0]),
                             upper=rvec[2][0] + 0.3 * np.abs(rvec[2][0]))
        t1 = ng.p.Scalar(init=tvec[0][0],
                         lower=tvec[0][0] - 0.3 * np.abs(tvec[0][0]),
                         upper=tvec[0][0] + 0.3 * np.abs(tvec[0][0]))
        t2 = ng.p.Scalar(init=tvec[1][0],
                         lower=tvec[1][0] - 0.3 * np.abs(tvec[1][0]),
                         upper=tvec[1][0] + 0.3 * np.abs(tvec[1][0]))
        t3 = ng.p.Scalar(init=tvec[2][0],
                         lower=tvec[2][0] - 0.3 * np.abs(tvec[2][0]),
                         upper=tvec[2][0] + 0.3 * np.abs(tvec[2][0]))
        f = ng.p.Scalar(init=f,
                        lower=f - 0.3 * f,
                        upper=f + 0.3 * f)
        params = ng.p.Instrumentation(theta1, theta2, theta3, t1, t2, t3, f)
        optimizer = ng.optimizers.registry[self.opt_name](parametrization=params,
                                                          budget=self.budget_2,
                                                          num_workers=5)
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(self.get_finetune_loss_cma,
                                                executor=executor, batch_mode=False)
        res = recommendation.args
        return np.array([[res[0]], [res[1]], [res[2]]]), np.array([[res[3]], [res[4]], [res[5]]]), res[6]

    def preliminary(self, object_points_dict, image_points_dict, vmm_results):
        self.object_points_full = []
        self.image_points_full = []
        for i, each in enumerate(image_points_dict):
            temp1, temp2 = self.prepare_keypoint_pairs(object_points_dict[vmm_results[str(i)]["model"]], each)
            self.object_points_full.append(temp1)
            self.image_points_full.append(temp2)

    def run(self):
        print("Parameter Finetuning...")
        object_points_dict = load_object_points(self.object_points_path)
        image_points_dict = load_image_points(self.image_points_dir)
        vmm_results = self.load_vmm_results()
        self.preliminary(object_points_dict, image_points_dict, vmm_results)

        images = []
        res = {}
        loss_old_min = 1e5
        loss_new_min = 1e5
        best_old_index = -1
        best_new_index = -1
        best_rvec = None
        best_tvec = None
        best_f = None

        for i, each in enumerate(image_points_dict):
            self.cma_object_points, self.cma_image_points = self.prepare_keypoint_pairs(object_points_dict[vmm_results[str(i)]["model"]], each)
            self.fixed_object_points = self.cma_object_points
            rvec_old = np.array(vmm_results[str(i)]["rvec"])
            tvec_old = np.array(vmm_results[str(i)]["tvec"])
            f_old = np.array(vmm_results[str(i)]["f"])
            loss_old = self.get_finetune_loss(self.fixed_object_points, rvec_old, tvec_old, f_old)[0]
            rvec_new, tvec_new, f_new = self.optimize(rvec_old, tvec_old, f_old)
            loss_new = self.get_finetune_loss(self.fixed_object_points, rvec_new, tvec_new, f_new)[0]
            if loss_old < loss_old_min:
                loss_old_min = loss_old
                best_old_index = i

            if loss_new < loss_new_min:
                loss_new_min = loss_new
                best_new_index = i
                best_rvec = rvec_new
                best_tvec = tvec_new
                best_f = f_new

            if self.verbose:
                print("Before optimizing:")
                print("\t Vehicle %02d, finetune loss %f" % (i, loss_old))
                print("After optimizing:")
                print("\t Vehicle %02d, finetune loss %f" % (i, loss_new))

        if self.verbose:
            print("Before vehicle model matching:")
            print("\t Vehicle %02d, finetune loss %f" % (best_old_index, loss_old_min))
            print("After vehicle model matching:")
            print("\t Vehicle %02d, finetune loss %f" % (best_new_index, loss_new_min))

        images.append(self.visualization(best_rvec, best_tvec, best_f,
                                         best_new_index,
                                         vmm_results[str(best_new_index)]["model"]))
        res["rvec"] = best_rvec.tolist()
        res["tvec"] = best_tvec.tolist()
        res["f"] = best_f
        res["loss"] = loss_new_min.tolist()

        self.create_montage(images, os.path.join(self.save_dir, "pf_montage.jpg"), 1, 1)
        with open(os.path.join(self.save_dir, "pf_results.json"), "w") as f:
            json.dump(res, f)

if __name__ == "__main__":
    config = {}
    config["camera_code"] = "K109F"
    config["image_height"] = 240
    config["image_width"] = 320
    config["focal_length"] = 350
    config["alpha"] = 1
    config["temperature"] = 0
    config["budget_1"] = 2000
    config["budget_2"] = 20000
    config["use_ransac"] = False
    config["opt_name"] = "CMA"

    config["object_points_path"] = "../examples/camera_calibration/K109F/objects/vehicle_3dkeypoints_head.json"
    config["image_points_dir"] = "../examples/camera_calibration/K109F/labels"
    config["images_dir"] = "../examples/camera_calibration/K109F/images"
    config["save_dir"] = "../examples/camera_calibration/K109F/outputs"

    config["verbose"] = True
    cg = ParameterFinetune(config)
    cg.run()