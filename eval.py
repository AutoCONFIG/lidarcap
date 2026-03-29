from tools import util
import metric
import sys


def eval(name):
    """评估模型性能
    
    Args:
        name: 模型文件名或路径
    """
    if not name:
        print("[ERROR] 请提供模型名称")
        return
    
    print(f"[INFO] 评估模型: {name}")
    
    for idx in [7, 24, 29, 41]:
        try:
            pred_poses = util.get_pred_poses(name, idx)
            gt_poses = util.get_gt_poses(idx)
            pred_poses = pred_poses[:len(gt_poses)]
            metric.output_metric(pred_poses, gt_poses)
        except FileNotFoundError as e:
            print(f"[WARNING] 数据集 {idx} 未找到: {e}")
        except Exception as e:
            print(f"[ERROR] 评估数据集 {idx} 时出错: {e}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = input("请输入模型名称: ").strip()
    
    eval(name)
