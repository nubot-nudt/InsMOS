#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<cmath>
#include<omp.h>

namespace py = pybind11;

#include<pybind11/numpy.h>


//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 idx
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
        // get bbox center
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
            
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
      
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
        // 
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
            //
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
            //
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


PYBIND11_MODULE(Array_Index, m)
{
    // 
    m.doc() = "pybind11 example plugin";
    
    m.def("find_features_by_bbox_with_yaw", &find_features_by_bbox_with_yaw, "A function return array idx other array ");
    
    m.def("find_point_in_instance_bbox_with_yaw", &find_point_in_instance_bbox_with_yaw, "A function return array idx other array ");
}




