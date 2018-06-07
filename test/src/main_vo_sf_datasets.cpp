/*********************************************************************************
**Fast Odometry and Scene Flow from RGB-D Cameras based on Geometric Clustering	**
**------------------------------------------------------------------------------**
**																				**
**	Copyright(c) 2017, Mariano Jaimez Tarifa, University of Malaga & TU Munich	**
**	Copyright(c) 2017, Christian Kerl, TU Munich								**
**	Copyright(c) 2017, MAPIR group, University of Malaga						**
**	Copyright(c) 2017, Computer Vision group, TU Munich							**
**																				**
**  This program is free software: you can redistribute it and/or modify		**
**  it under the terms of the GNU General Public License (version 3) as			**
**	published by the Free Software Foundation.									**
**																				**
**  This program is distributed in the hope that it will be useful, but			**
**	WITHOUT ANY WARRANTY; without even the implied warranty of					**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the				**
**  GNU General Public License for more details.								**
**																				**
**  You should have received a copy of the GNU General Public License			**
**  along with this program. If not, see <http://www.gnu.org/licenses/>.		**
**																				**
*********************************************************************************/

#include <stdio.h>
#include <joint_vo_sf.h>
#include <datasets.h>
#include <kmeans.h>
#include <visualization.h>
#include <segmentation_background.h>
#include <general.h>



int main()
{	
    const bool save_results = false;
    	unsigned int res_factor = 2;

    VO_SF cf(res_factor);//最基础的类，声明了很多变量～
    Datasets dataset(res_factor);//这个类主要用来载入测试的数据集
    K_means ks(res_factor,cf);//使用K-means来处理聚类问题
    seg_back sb(res_factor,cf,ks);
    visualization vl(res_factor,ks,sb,cf);
    general gl(cf,ks,sb,vl);

    //设置 Rawlog 文件的路径
    dataset.filename = "/home/payne/文档/Joint-VO-SF-master/data/rawlog_rgbd_dataset_freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz.rawlog";

    //初始化3D显示场景
    vl.initializeSceneDatasets();


    if (save_results)
        dataset.CreateResultsFile();//如果保存文件，先创建相应文件
    dataset.openRawlog();//打开Rawlog文件，并载入相关信息
    dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.intensity_wf, cf.im_r, cf.im_g, cf.im_b);//载入Rawlog中每一帧的深度图像和RGB图像
   cf.cam_pose = dataset.gt_pose; cf.cam_oldpose = dataset.gt_pose;
    cf.createImagePyramid();//对每一帧图像构造图像金字塔

    int  stop = 0;
    bool anything_new = false;

    while (!stop)
    {
            dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.intensity_wf, cf.im_r, cf.im_g, cf.im_b);//载入Rawlog中每一帧的深度图像和RGB图像
            gl.run_VO_SF(true);//核心
            vl.createImagesOfSegmentations();//利用得到的b_segm来标记出图像中的动态物体
            if (save_results)
                dataset.writeTrajectoryFile(cf.cam_pose, cf.ddt);//保存轨迹文件
            anything_new = 1;

        if (anything_new)
        {
            bool aux = false;
            vl.updateSceneDatasets(dataset.gt_pose, dataset.gt_oldpose);//更新可视化内容
            anything_new = 0;
        }
    }


    return 0;
}

