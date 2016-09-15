#ifndef UTILITIES3D_HPP_INCLUDED
#define UTILITIES3D_HPP_INCLUDED

#include <iostream>
#include <string>
#include <fstream>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
// PCL 
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <pcl/features/shot_omp.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfh.h>
#include <pcl/features/range_image_border_extractor.h>
// BOOST
#include "boost/variant.hpp"
#include <boost/variant/get.hpp>
#include <boost/thread/thread.hpp>

//Funcion que escoge el descriptor arrojado por el variant de boost
//segun su tipo de dato
#define DescriptorByType boost::get<typename pcl::PointCloud<Type>::Ptr>(cloudNow.getDescriptors(descriptor)) 

using namespace cv;
using namespace std;
using namespace cv_bridge;
using namespace pcl;


typedef pcl::PointCloud<pcl::PointXYZ> PCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PtrPCloud;
typedef pcl::PointCloud <pcl::PointXYZRGB> PCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PtrPCloudRGB;
typedef pcl::PointCloud<pcl::Normal>::Ptr PtrNormal;


typedef pcl::PointCloud<pcl::SHOT352>::Ptr DescriptorSHOT; 
typedef pcl::PointCloud<pcl::FPFHSignature33>::Ptr DescriptorFPFH;
typedef pcl::PointCloud<pcl::Narf36>::Ptr DescriptorNARF;
typedef pcl::PointCloud<pcl::PFHSignature125>::Ptr DescriptorPFH;


typedef boost::variant<DescriptorFPFH, DescriptorSHOT, DescriptorNARF, DescriptorPFH> DescVar;
typedef boost::shared_ptr<Correspondences> PtrCorrespondences;

	 
enum PCL_DESCRIPTORS{

    F_FPFH = 0,
    F_PFH = 1,   
    F_SHOT =2,
    F_NARF =3
};

class Results{

	public:

	int goodMatches;
	int badMatches;
	float tiempo;

	Results(){
		goodMatches = 0;
		badMatches = 0;
		tiempo = 0;
	}
	float indiceAcierto(){
		float total  = goodMatches+badMatches;
		if(total!=0)
			return (goodMatches*100)/(total);
		return 0;
	}
	//Imprime el resultado del metodo detector y descriptor usados 
	string print(string detector, string descriptor){
		stringstream ss;
		float porcentaje = floorf(indiceAcierto() * 100 + 0.5) / 100;
		ss << "---------------"<<endl;
		ss <<"| " << detector << " + " << descriptor << " | "; 
		ss << "Rating: " << porcentaje/tiempo << endl;
		ss << "---------------"<<endl;
		ss <<"| " << "GoodMatches: "<< goodMatches << " | ";
		ss << "BadMatches: "<< badMatches << " | ";
		
		ss << "Accuracy: " << porcentaje <<"%" << " | ";
		ss << "Time: "<< tiempo << " s"<<" | "<< endl<<endl;
		 
		return ss.str();
	}

};

class CloudContainer{
		
	public:

	//Caracteristicas segun el metodo
	//Vector que se corresponde del modo indice = metodo
	//Y en cada metodo se asocia un par keypoints, descriptor
	PtrPCloudRGB keypoints;
	DescriptorNARF narfDescriptors;
	DescriptorFPFH fpfhDescriptors;
	DescriptorSHOT shotDescriptors;
	DescriptorPFH pfhDescriptors;

	PtrPCloudRGB cloud;
	PtrPCloudRGB tranformCloud;


	CloudContainer & operator=(const CloudContainer &c){
		if(&c==this)
			return *this;
		else{
			
			keypoints = c.keypoints;
			narfDescriptors = c.narfDescriptors;
			fpfhDescriptors = c.fpfhDescriptors;
			shotDescriptors = c.shotDescriptors;
			pfhDescriptors = c.pfhDescriptors;
			cloud = c.cloud;
			tranformCloud = c.tranformCloud;	
		}
		return *this;
	}
	DescVar getDescriptors(int descriptor){

		switch (descriptor){
	  		case F_FPFH:  
		  		return fpfhDescriptors;
		  	case F_SHOT:
		  		return shotDescriptors;
		  	case F_NARF:
	  			return narfDescriptors;
	  		case F_PFH:
	  			return pfhDescriptors;
		}
	}

};


void removeNaN(PtrPCloudRGB &cloud){
	vector<int> indices;
	PtrPCloudRGB cloudTemp(new PCloudRGB);
	pcl::removeNaNFromPointCloud(*cloud, *cloudTemp, indices);

	cloud = cloudTemp;
}


PtrPCloudRGB getCloudColorDepth(const Mat imageColor, const float* depthImage){

    PtrPCloudRGB cloud(new PCloudRGB);
    cloud->height = 480;
    cloud->width = 640;
    cloud->is_dense = false;

    cloud->points.resize(cloud->height * cloud->width);

    register float constant = 0.0019047619;
    cloud->header.frame_id = "/openni_rgb_optical_frame";

    register int centerX = (cloud->width >> 1);
    int centerY = (cloud->height >> 1);

    float bad_point = std::numeric_limits<float>::quiet_NaN();

    register int depth_idx = 0;
    int i,j;
    for (int v = -centerY,j=0; v < centerY; ++v,++j)
    {
        for (register int u = -centerX,i=0; u < centerX; ++u, ++depth_idx,++i)
        {
			pcl::PointXYZRGB& pt = cloud->points[depth_idx];

	        float depthimagevalue=depthImage[depth_idx];

	        if (depthimagevalue == 0){
		        // not valid
		        pt.x = pt.y = pt.z = bad_point;
		        continue;
	        }
			pt.z = depthimagevalue;
			pt.x = u * pt.z * constant;
			pt.y = v * pt.z * constant;

	        const Point3_<uchar>* p = imageColor.ptr<Point3_<uchar> >(j,i);
	        pt.r=p->z;
	        pt.g=p->y;
	        pt.b=p->x;
        }
      }
    return cloud;
}

void getRangeImage(RangeImage& range_image, const PtrPCloudRGB &cloud){

	float noise_level = 0.0;
	float min_range = 0.0f;
	int border_size = 1;
	float angular_resolution = 0.5f;
	angular_resolution = pcl::deg2rad (angular_resolution);
	RangeImage::CoordinateFrame coordinate_frame = RangeImage::CAMERA_FRAME;

	Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
	boost::shared_ptr<RangeImage> range_image_ptr (new RangeImage);
	range_image = *range_image_ptr;

	scene_sensor_pose =
	Eigen::Affine3f (Eigen::Translation3f (cloud->sensor_origin_[0], cloud->sensor_origin_[1], cloud->sensor_origin_[2])) * Eigen::Affine3f (cloud->sensor_orientation_);


	range_image.createFromPointCloud ((*cloud),  angular_resolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
}


//Extracción de keypoints mediante NARF
pcl::PointCloud<int> applyNARF_Key(CloudContainer &cloudCont, RangeImage& range_image){

    pcl::PointCloud<int> keypoint_indices;  //detectores
    PtrPCloudRGB keypoints_narf( new PCloudRGB);
    pcl::PointCloud<pcl::Narf36>::Ptr narf_descriptors( new pcl::PointCloud<pcl::Narf36> );
   
    pcl::RangeImageBorderExtractor range_image_border_extractor;
    pcl::NarfKeypoint narf_keypoint_detector;

    narf_keypoint_detector.setRangeImageBorderExtractor(&range_image_border_extractor);
    narf_keypoint_detector.setRangeImage (&range_image);
    narf_keypoint_detector.getParameters ().support_size = 0.5f;
    narf_keypoint_detector.setRadiusSearch(0.5);                   
    narf_keypoint_detector.compute (keypoint_indices);

    keypoints_narf->width = keypoint_indices.points.size();
    keypoints_narf->height = 1;
    keypoints_narf->is_dense = false;
    keypoints_narf->points.resize (keypoints_narf->width * keypoints_narf->height);
           
    int ind_count=0;
    for (size_t i = 0; i < keypoint_indices.points.size(); i++){
        ind_count = keypoint_indices.points[i];
                           
        keypoints_narf->points[i].x = range_image.points[ind_count].x;
        keypoints_narf->points[i].y = range_image.points[ind_count].y;
        keypoints_narf->points[i].z = range_image.points[ind_count].z;
    }

    removeNaN(keypoints_narf);

    cloudCont.keypoints = keypoints_narf;

    return keypoint_indices;

}

double computeCloudResolution (const PtrPCloudRGB &cloud){

  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<pcl::PointXYZRGB> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

// -----------  DETECTORES ----------- //


	

void applyHarris(CloudContainer &cloudCont){
    pcl::PointCloud<pcl::PointXYZI>::Ptr harris_keypoints( new pcl::PointCloud<pcl::PointXYZI>);
    PtrPCloudRGB keyPoints(new PCloudRGB);
    pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI,pcl::Normal >::Ptr harris_detector(new pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI,pcl::Normal >);

    harris_detector->setNonMaxSupression(true);
    harris_detector->setSearchSurface (cloudCont.cloud);
    harris_detector->setRadius(0.009f);
    harris_detector->setRadiusSearch(0.009f);
    harris_detector->setThreshold(0.001f);
    harris_detector->setInputCloud (cloudCont.cloud);

    harris_detector->setNumberOfThreads (4);
    harris_detector->compute (*harris_keypoints);

    pcl::copyPointCloud(*harris_keypoints,*keyPoints);

    cloudCont.keypoints = keyPoints;
}

void applySIFT(CloudContainer &cloudCont){

	const float min_scale = 0.01;
	const int nr_octaves = 2;
	const int nr_scales_per_octave = 3;
	const float min_contrast = 1;
	const PtrPCloudRGB src= cloudCont.cloud;
	PtrPCloudRGB keypoints_src (new PCloudRGB);
	pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_detect;

    if(src->isOrganized() ){
         pcl::search::OrganizedNeighbor<pcl::PointXYZRGB>::Ptr on(new pcl::search::OrganizedNeighbor<pcl::PointXYZRGB>());
         sift_detect.setSearchMethod(on);
    }else{
         pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
         sift_detect.setSearchMethod(tree);
    }
    sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
    sift_detect.setMinimumContrast (min_contrast);

    sift_detect.setInputCloud (src);
    pcl::PointCloud<pcl::PointWithScale> keypoints_temp;
    sift_detect.compute (keypoints_temp);

    pcl::copyPointCloud (keypoints_temp,*keypoints_src);

    cloudCont.keypoints = keypoints_src;
}

//Extractor Keypoints ISS3D
void applyISS(CloudContainer &cloudCont){

	PtrPCloudRGB model_keypoints (new PCloudRGB);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
	const PtrPCloudRGB src = cloudCont.cloud;
	std::vector<int> indices1;
	//
	//  ISS3D parameters
	//
	double iss_salient_radius_;
	double iss_non_max_radius_;
	double iss_normal_radius_;
	double iss_border_radius_;
	double iss_gamma_21_ (0.575);
	double iss_gamma_32_ (0.575);
	double iss_min_neighbors_ (5);
	int iss_threads_ (4);

	// Fill in the model cloud

	double model_resolution= computeCloudResolution(cloudCont.cloud);

	// Compute model_resolution

	iss_salient_radius_ = 6 * model_resolution;
	iss_non_max_radius_ = 4 * model_resolution;
	iss_normal_radius_ = 4 * model_resolution;
	iss_border_radius_ = 1 * model_resolution;
	//
	// Compute keypoints
	//
	pcl::ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss_detector;

	iss_detector.setSearchMethod (tree);
	iss_detector.setSalientRadius (iss_salient_radius_);
	iss_detector.setNonMaxRadius (iss_non_max_radius_);

	iss_detector.setNormalRadius (iss_normal_radius_);
	iss_detector.setBorderRadius (iss_border_radius_);

	iss_detector.setThreshold21 (iss_gamma_21_);
	iss_detector.setThreshold32 (iss_gamma_32_);
	iss_detector.setMinNeighbors (iss_min_neighbors_);
	iss_detector.setNumberOfThreads (iss_threads_);
	iss_detector.setInputCloud (src);
	iss_detector.compute (*model_keypoints);

	cloudCont.keypoints = model_keypoints;
}
void applyNARF_DESC(pcl::PointCloud<int> keypoint_indices,const RangeImage& range_image, CloudContainer &cloudCont){

	pcl::NarfKeypoint narf_keypoint_detector;
	vector<int> feature_indice;

	//******NARF DESCRIPTORES******//
	
	feature_indice.resize (keypoint_indices.points.size ());
	for (unsigned int i=0; i<keypoint_indices.size (); ++i){ 
		feature_indice[i]=keypoint_indices.points[i];
	}
	pcl::NarfDescriptor narf_descriptor (&range_image, &feature_indice);
	narf_descriptor.getParameters ().support_size = 0.02f;
	narf_descriptor.getParameters ().rotation_invariant = true;
	pcl::PointCloud<pcl::Narf36>::Ptr narfDescriptors(new pcl::PointCloud<pcl::Narf36>);

	narf_descriptor.compute (*narfDescriptors);
	

	cloudCont.narfDescriptors = narfDescriptors;
	//cout << "Extracted "<<narfDescriptors->size ()<<" descriptors for "<<keypoint_indices.points.size ()<< " keypoints.\n";

}
void getNormal(PtrPCloudRGB &keypoints, PtrNormal &cloud_normals){
             
    pcl::NormalEstimationOMP<PointXYZRGB,pcl::Normal> nest;
    nest.setInputCloud (keypoints);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    nest.setSearchMethod (tree);
    nest.setNumberOfThreads(4);
    nest.setRadiusSearch (0.4);

    nest.compute (*cloud_normals);
}

//----------- DESCRIPTORES -----------//

/*
	Para anyadir nuevos descriptores hay que:
	- Anyadir un nuevo tipo de devolucion para el boost:variant
	- Anyadir entrada en el switch de la funcion getDescriptors
	- Anyadir entrada en el switch de la funcion useMethod
	- Operator= CloudContainer
*/

void applyFPFH(CloudContainer &cloudCont){

	DescriptorFPFH fpfh_src (new pcl::PointCloud<pcl::FPFHSignature33>);
    PtrNormal cloud_normals (new pcl::PointCloud<pcl::Normal>);
    getNormal(cloudCont.keypoints, cloud_normals);

	FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh_est_src;

	fpfh_est_src.setRadiusSearch (0.1);
    fpfh_est_src.setNumberOfThreads(4);
    fpfh_est_src.setInputCloud (cloudCont.keypoints);
    fpfh_est_src.setInputNormals (cloud_normals);
    //PCL_INFO (" FPFH - Compute Source\n");
    fpfh_est_src.compute (*fpfh_src);
    //PCL_INFO("FPFH - Size of descriptors %d\n",fpfh_src->size());

    cloudCont.fpfhDescriptors = fpfh_src;
}
void applySHOT(CloudContainer &cloudCont){

    DescriptorSHOT descriptorSHOT (new pcl::PointCloud<pcl::SHOT352>);
    PtrNormal cloud_normals (new pcl::PointCloud<pcl::Normal>);
    getNormal(cloudCont.keypoints, cloud_normals);

    pcl::SHOTEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> descr_est;

    descr_est.setRadiusSearch (0.2f);
    descr_est.setNumberOfThreads(4);
    descr_est.setLRFRadius(1.8f);
    descr_est.setInputCloud (cloudCont.keypoints);
    descr_est.setInputNormals (cloud_normals);
    descr_est.setSearchSurface (cloudCont.keypoints);

    descr_est.compute (*descriptorSHOT);

    cloudCont.shotDescriptors = descriptorSHOT;

}

void applyPFH(CloudContainer &cloudCont){
        
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHSignature125> pfh_est_src;
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_src (new pcl::PointCloud<pcl::PFHSignature125>);
    getNormal(cloudCont.keypoints,cloud_normals);
   
    pfh_est_src.setInputCloud (cloudCont.keypoints);
    pfh_est_src.setInputNormals (cloud_normals);
    pfh_est_src.setRadiusSearch (0.05);

    pfh_est_src.compute (*pfh_src);

    cloudCont.pfhDescriptors = pfh_src;
}



#endif