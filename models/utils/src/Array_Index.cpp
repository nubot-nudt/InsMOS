#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<cmath>
#include<omp.h>

namespace py = pybind11;

#include<pybind11/numpy.h>

//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
std::vector<int> find_array_idx_bybbox(py::array_t<int>& arr1, py::array_t<float>& arr2) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    
    std::vector<int> index_arr1;
    int center[3];
    int extend[3];
    for (int i=0;i<arr2.shape()[0];i++)
    {
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2);
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        for(int j=0;j<arr1.shape()[0];j++)
        {
            if((r1(j,0)<=(center[0]+extend[0]/2+1))&&(r1(j,0)>=(center[0]-extend[0]/2-1)) \
                &&(r1(j,1)<(center[1]+extend[1]/2+1))&&(r1(j,1)>(center[1]-extend[1]/2-1)) \
                    &&(r1(j,2)<(center[2]+extend[2]/2))&&(r1(j,2)>(center[2]-extend[2]/2)))
            {
                index_arr1.push_back(j);
            }
        }
    }
    return index_arr1;
}

//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
py::array_t<int> find_features_by_bbox(py::array_t<int>& arr1, py::array_t<float>& arr2, py::array_t<int>& arr3) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    auto r3 = arr3.mutable_unchecked<2>();

    omp_set_num_threads(30);
    #pragma omp parallel for
    for (int i=0;i<arr2.shape()[0];i++)
    {
        float center[3];
        float extend[3];
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2);
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        
        for(int j=0;j<arr1.shape()[0];j++)
        {
            if((r1(j,0)<=(center[0]+extend[0]/2+1))&&(r1(j,0)>=(center[0]-extend[0]/2-1)) \
                &&(r1(j,1)<(center[1]+extend[1]/2+1))&&(r1(j,1)>(center[1]-extend[1]/2-1)) \
                    &&(r1(j,2)<(center[2]+extend[2]/2+1))&&(r1(j,2)>(center[2]-extend[2]/2)))
            {
                if(int(r2(i,7))>0)
                {
                    r3(j,int(r2(i,7))-1)=1;
                }
            }
        }
    }
    return arr3;
}

//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
py::array_t<int> find_features_by_bbox_with_yaw(py::array_t<int>& arr1, py::array_t<float>& arr2, py::array_t<int>& arr3) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    auto r3 = arr3.mutable_unchecked<2>();

    unsigned int arr1_length=arr1.shape()[0];
    unsigned int arr2_length=arr2.shape()[0];
    omp_set_num_threads(30);
    #pragma omp parallel for
    for (int i=0;i<arr2_length;i++)
    {
        float center[3];
        float extend[3];
        float theta;
        float centered[3];
        float rotated_point[3];
        int first_point[3];
        char first_flag;
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2);
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        theta = r2(i,6);
        float  cos_theta = cos(theta);
        float sin_theta = sin(theta);
        first_flag=0;
        // omp_set_num_threads(30);
        // #pragma omp parallel for
        for(int j=0;j<arr1_length;j++)
        {

            if((first_flag==1)&&(r1(j,0)>(first_point[0]+extend[0])||r1(j,0)<(first_point[0]-extend[0])||r1(j,1)>(first_point[1]+extend[1])||r1(j,1)<(first_point[1]-extend[1])||r1(j,2)>(first_point[2]+extend[2])||r1(j,2)<(first_point[2]-extend[2])))
            {
                    continue;
            }
            //去中心化，让这个bbox为中心
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
            //乘以旋转矩阵
            rotated_point[0] = centered[0]*cos_theta + centered[1]*sin_theta;
            rotated_point[1] = -centered[0]*sin_theta+ centered[1]*cos_theta;
            if((rotated_point[0]<=extend[0]/2)&&(rotated_point[0]>=-extend[0]/2) \
                &&(rotated_point[1]<=extend[1]/2)&&(rotated_point[1]>=-extend[1]/2) \
                    &&(centered[2]<=extend[2]/2)&&(centered[2]>=-extend[2]/2))
            {
                if(int(r2(i,7))>0)
                {
                    r3(j,int(r2(i,7))-1)=1;
                }
                if(first_flag==0)
                {
                    first_flag=1;
                    first_point[0] = r1(j,0);
                    first_point[1] = r1(j,1);
                    first_point[2] = r1(j,2);
                    
                }
            }
        }
    }
    return arr3;
}
//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
py::array_t<int> find_features_by_bbox_in_point_with_yaw(py::array_t<float>& arr1, py::array_t<float>& arr2, py::array_t<int>& arr3) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    auto r3 = arr3.mutable_unchecked<2>();

    unsigned int arr1_length=arr1.shape()[0];
    unsigned int arr2_length=arr2.shape()[0];
    omp_set_num_threads(30);
    #pragma omp parallel for
    for (int i=0;i<arr2_length;i++)
    {
        float center[3];
        float extend[3];
        float theta;
        float centered[3];
        float rotated_point[3];
        float first_point[3];
        char first_flag;
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2);
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        theta = r2(i,6);
        float  cos_theta = cos(theta);
        float sin_theta = sin(theta);
        first_flag=0;
        // omp_set_num_threads(30);
        // #pragma omp parallel for
        for(int j=0;j<arr1_length;j++)
        {

            if((first_flag==1)&&(r1(j,0)>(first_point[0]+extend[0])||r1(j,0)<(first_point[0]-extend[0])||r1(j,1)>(first_point[1]+extend[1])||r1(j,1)<(first_point[1]-extend[1])||r1(j,2)>(first_point[2]+extend[2])||r1(j,2)<(first_point[2]-extend[2])))
            {
                    continue;
            }
            //去中心化，让这个bbox为中心
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
            //乘以旋转矩阵
            rotated_point[0] = centered[0]*cos_theta + centered[1]*sin_theta;
            rotated_point[1] = -centered[0]*sin_theta+ centered[1]*cos_theta;
            if((rotated_point[0]<=extend[0]/2)&&(rotated_point[0]>=-extend[0]/2) \
                &&(rotated_point[1]<=extend[1]/2)&&(rotated_point[1]>=-extend[1]/2) \
                    &&(centered[2]<=extend[2]/2)&&(centered[2]>=-extend[2]/2))
            {
                if(int(r2(i,7))>0)
                {
                    r3(j,int(r2(i,7))-1)=1;
                }
                if(first_flag==0)
                {
                    first_flag=1;
                    first_point[0] = r1(j,0);
                    first_point[1] = r1(j,1);
                    first_point[2] = r1(j,2);
                    
                }
            }
        }
    }
    return arr3;
}

//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
py::array_t<int> find_point_in_instance_bbox_with_yaw(py::array_t<float>& arr1, py::array_t<float>& arr2, py::array_t<int>& arr3,float out_ground) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    auto r3 = arr3.mutable_unchecked<2>();

    unsigned int arr1_length=arr1.shape()[0];
    unsigned int arr2_length=arr2.shape()[0];
    omp_set_num_threads(30);
    #pragma omp parallel for
    for (int i=0;i<arr2_length;i++)
    {
        float center[3];
        float extend[3];
        float theta;
        float centered[3];
        float rotated_point[3];
        float first_point[3];
        char first_flag;
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2)+out_ground;
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        theta = r2(i,6);
        float  cos_theta = cos(theta);
        float sin_theta = sin(theta);
        first_flag=0;
        // if(center[0]<5 &&center[0]>-5 && center[1]<5 && center[1]>-5)
        // {
        //     out_ground = 0;
        // }
        // omp_set_num_threads(30);
        // #pragma omp parallel for
        for(int j=0;j<arr1_length;j++)
        {

            if((first_flag==1)&&(r1(j,0)>(first_point[0]+extend[0])||r1(j,0)<(first_point[0]-extend[0])||r1(j,1)>(first_point[1]+extend[1])||r1(j,1)<(first_point[1]-extend[1])||r1(j,2)>(first_point[2]+extend[2])||r1(j,2)<(first_point[2]-extend[2])))
            {
                    continue;
            }
            //去中心化，让这个bbox为中心
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
            //乘以旋转矩阵
            rotated_point[0] = centered[0]*cos_theta + centered[1]*sin_theta;
            rotated_point[1] = -centered[0]*sin_theta+ centered[1]*cos_theta;
            if((rotated_point[0]<=extend[0]/2)&&(rotated_point[0]>=-extend[0]/2) \
                &&(rotated_point[1]<=extend[1]/2)&&(rotated_point[1]>=-extend[1]/2) \
                    &&(centered[2]<=extend[2]/2)&&(centered[2]>=-extend[2]/2))
            {
                if(int(r2(i,7))>0)
                {
                    r3(j,int(r2(i,7))-1)=i+1;
                }
                if(first_flag==0)
                {
                    first_flag=1;
                    first_point[0] = r1(j,0);
                    first_point[1] = r1(j,1);
                    first_point[2] = r1(j,2);
                    
                }
            }
        }
    }
    return arr3;
}
//*2d find poin in boundingboox with yaw and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
std::vector<int> find_array_idx_bybbox_with_yaw(py::array_t<int>& arr1, py::array_t<float>& arr2) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    std::vector<int> index_arr1;
    int center[3];
    int extend[3];
    float theta;
    int centered[3];
    int rotated_point[3];
    int first_point[3];
    char first_flag;
    for (int i=0;i<arr2.shape()[0];i++)
    {
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2);
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        
        theta = r2(i,6);
        first_flag=0;
        for(int j=0;j<arr1.shape()[0];j++)
        {
            if((first_flag==1)&&(r1(j,0)>(first_point[0]+extend[0])||r1(j,0)<(first_point[0]-extend[0])||r1(j,1)>(first_point[1]+extend[1])||r1(j,1)<(first_point[1]-extend[1])||r1(j,2)>(first_point[2]+extend[2])||r1(j,2)<(first_point[2]-extend[2])))
            {
                    continue;
            }
            //去中心化，让这个bbox为中心
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
            //乘以旋转矩阵
            rotated_point[0] = centered[0]*cos(theta) + centered[1]*sin(theta);
            rotated_point[1] = -centered[0]*sin(theta)+ centered[1]*cos(theta);
            rotated_point[2] = centered[2];
            if((rotated_point[0]<=extend[0]/2)&&(rotated_point[0]>=-extend[0]/2) \
                &&(rotated_point[1]<=extend[1]/2)&&(rotated_point[1]>=-extend[1]/2) \
                    &&(rotated_point[2]<=extend[2]/2)&&(rotated_point[2]>=-extend[2]/2))
            {
                index_arr1.push_back(j);
                if(first_flag==0)
                {
                    first_flag=1;
                    first_point[0] = r1(j,0);
                    first_point[1] = r1(j,1);
                    first_point[2] = r1(j,2);
                }
            }
        }
    }
    return index_arr1;
}
//*2d find poin in other point and return idx 
//*arr1 point (N,3)
//*arr2 point (N,3)
py::array_t<int> find_point_in_other_point_idx(py::array_t<float>& arr1, py::array_t<float>& arr2, py::array_t<int>& arr3) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    auto r3 = arr3.mutable_unchecked<2>();

    unsigned int arr1_length=arr1.shape()[0];
    unsigned int arr2_length=arr2.shape()[0];
    // omp_set_num_threads(10);
    // #pragma omp parallel for
    #pragma omp parallel for num_threads(10)
    for (int i=0;i<arr2_length;i++)
    {
        for(int j=0;j<arr1_length;j++)
        {
            if(r1(j,0)==r2(i,0))
            {
                if(r1(j,1)==r2(i,1) && r1(j,2)==r2(i,2))
                {
                    r3(i,0) = j;
                    break;
                }
            }
        }
    }
    return arr3;
}
std::vector<int> find_array_idx(py::array_t<int>& arr1, py::array_t<int>& arr2) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    std::vector<int> index_arr1;
    for (int i = 0; i < arr1.shape()[0]; i++) {
        for (int j = 0; j < arr2.shape()[0]; j++) {
            if (r1(i,0)==r2(j,0) && r1(i,1)==r2(j,1)  && r1(i,2)==r2(j,2))
            {
                index_arr1.push_back(i);
                break;
            }
        } 
    }
    return index_arr1;
}

PYBIND11_MODULE(Array_Index, m)
{
    // 可选，说明这个模块是做什么的
    m.doc() = "pybind11 example plugin";
    //def( "给python调用方法名"， &实际操作的函数， "函数功能说明" ). 其中函数功能说明为可选
    m.def("find_features_by_bbox", &find_features_by_bbox, "A function return array idx other array ");
    m.def("find_features_by_bbox_with_yaw", &find_features_by_bbox_with_yaw, "A function return array idx other array ");

    m.def("find_features_by_bbox_in_point_with_yaw", &find_features_by_bbox_in_point_with_yaw, "A function return array idx other array ");
    m.def("find_point_in_instance_bbox_with_yaw", &find_point_in_instance_bbox_with_yaw, "A function return array idx other array ");

    m.def("find_point_in_other_point_idx", &find_point_in_other_point_idx, "A function return array idx other array ");

    m.def("find_array_idx_bybbox", &find_array_idx_bybbox, "A function return array idx other array ");
    m.def("find_array_idx_bybbox_with_yaw", &find_array_idx_bybbox_with_yaw, "A function return array idx other array ");
    m.def("find_array_idx", &find_array_idx, "A function return array idx other array ");
}




