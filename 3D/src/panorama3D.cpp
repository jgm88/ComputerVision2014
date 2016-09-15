#include "utilities3D.hpp"


using namespace cv;
using namespace std;
using namespace cv_bridge;
using namespace pcl;

//Globales que se ocupan de escribir los resultados en el fichero de salida
//muy a mi pesar es la unica manera de evitar lidiar con los problemas de variables
//y metodos estaticos al usar punteros a funcion
ofstream FICHERO;
string RESULTADOSFIN;
//tiempos inicio y fin de programa
struct timespec tStart, tEnd;

class Panorama3D{

	private:

	sensor_msgs::ImageConstPtr depthmsg;
	CvImagePtr imageColormsg;
	bool depthreceived, imagereceived;
	int iterProcess;

	//Ruta donde guardamos las capturas
	string rutaimagen;
	
	//objeto viewer
	pcl::visualization::PCLVisualizer::Ptr viewer;
	pcl::visualization::PCLVisualizer::Ptr viewerTransform;

	// Elementos para Suscriptor ImagenColor
	ros::NodeHandle nhColor_;
	image_transport::ImageTransport itColor_;
	image_transport::Subscriber imageColor_sub_;
	// Elementos para Suscriptor ImagenProfundidad
	ros::NodeHandle nhDepth_;
	image_transport::ImageTransport itDepth_;
	image_transport::Subscriber imageDepth_sub_;

	CloudContainer cloudLast, cloudNow;

	int nMethods;
	vector< Results > vResults;

	//Estructura para medir tiempos del algoritmo
	struct timespec ts1, ts2; 


	//Estructuras para crear el vector de combinaciones de metodos
	//Puntero a funcion que aplica combinacion 
	typedef void (Panorama3D::*ptrFunc)(string &,const int &);
	vector<ptrFunc> vMethods;

	Eigen::Matrix4f transformation;
	PtrPCloudRGB mergedCloud;

	ofstream myfile;

	public:



	Panorama3D():itColor_(nhColor_),itDepth_(nhDepth_), mergedCloud(new PCloudRGB){
		
		depthreceived = false;
		imagereceived = false;
		iterProcess = 0;

		viewer= pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("Matching"));
		viewer->setBackgroundColor (0, 0, 0);

		viewer->initCameraParameters ();
		viewer->setSize (800, 600);

		viewerTransform= pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("Transformation"));
		viewerTransform->setBackgroundColor (0, 0, 0);

		viewerTransform->initCameraParameters ();
		viewerTransform->setSize (800, 600);

		transformation.setIdentity();

		myfile.open ("traj_estimated.txt");

		//Controlamos la señal de corte de ejecucion para cerrar el fichero de resultados
		void (*handler)(int);
		handler = signal(SIGINT, cierraFichero);

  		FICHERO.open ("/home/javi/resultados.txt");
  		RESULTADOSFIN="";

  		//Anyadimos las combinaciones de metodos a probar
  		
  		vMethods.push_back(&Panorama3D::K_ISS_F_FPFH);
  		//vMethods.push_back(&Panorama3D::K_HARRYS_F_FPFH);
  		//vMethods.push_back(&Panorama3D::K_SIFT_F_FPFH);
  		
		//vMethods.push_back(&Panorama3D::K_ISS_F_SHOT);
  		//vMethods.push_back(&Panorama3D::K_HARRYS_F_SHOT);  		
  		//vMethods.push_back(&Panorama3D::K_SIFT_F_SHOT);  		
  		
  		//vMethods.push_back(&Panorama3D::K_ISS_F_PFH);
  		//vMethods.push_back(&Panorama3D::K_SIFT_F_PFH);
  		//vMethods.push_back(&Panorama3D::K_HARRYS_F_PFH);
  		  		
  		//OPCIONAL
  		//vMethods.push_back(&Panorama3D::K_NARF_F_NARF);

  		nMethods = vMethods.size();

  		vResults.resize(nMethods);

	imageColor_sub_ = itColor_.subscribe("/camera/rgb/image_color", 1, &Panorama3D::imageCb, this);
		imageDepth_sub_ = itDepth_.subscribe("/camera/depth/image", 1, &Panorama3D::imageCbdepth, this);


	}
	~Panorama3D(){
		myfile.close();
	}

	void showKeypoints(const PtrPCloudRGB &keypoints){

        PtrPCloudRGB keypoints_ptr (new PCloudRGB);
        PCloudRGB& keypointsTemp = *keypoints_ptr;

        keypointsTemp.points.resize (keypoints->points.size ());
        for (size_t i=0; i<keypoints->points.size (); ++i)
        keypointsTemp.points[i].getVector3fMap () = keypoints->points[i].getVector3fMap ();

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> keypoints_color_handler (keypoints, 0, 255, 0);
        viewer->addPointCloud<pcl::PointXYZRGB> (keypoints, keypoints_color_handler, "keypoints");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
	}


	void viewCorrespondences(const PtrCorrespondences &cor_all){

        PtrPCloudRGB cloud_ant= cloudLast.cloud;
		PtrPCloudRGB keypoint_ant= cloudLast.keypoints;
        Eigen::Affine3f transfrom_translation=pcl::getTransformation (5.0, 0, 0, 0, 0, 0);

        PtrPCloudRGB cloud_ant_transformed (new PCloudRGB);
        PtrPCloudRGB keyPoint_ant_transformed (new PCloudRGB);

        transformPointCloud (*cloud_ant, *cloud_ant_transformed,transfrom_translation);
        transformPointCloud (*keypoint_ant, *keyPoint_ant_transformed,transfrom_translation);

        visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud(cloudNow.cloud);
        if (!viewer->updatePointCloud (cloudNow.cloud,rgbcloud, "cloudn1")){
        	//intento actualizar la nube y si no existe la creo.
        	viewer->addPointCloud(cloudNow.cloud,rgbcloud,"cloudn1");
    	}

        visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud_ant(cloud_ant_transformed);
        if (!viewer->updatePointCloud (cloud_ant_transformed,rgbcloud_ant, "cloudn2")){
        	//intento actualizar la nube y si no existe la creo.
        	viewer->addPointCloud(cloud_ant_transformed,rgbcloud_ant,"cloudn2");
       	}

        string corresname="correspondences";
        if (!viewer->updateCorrespondences<pcl::PointXYZRGB>(cloudNow.keypoints,keyPoint_ant_transformed,*cor_all,1)) //intento actualizar la nube y si no existe la creo.
            viewer->addCorrespondences<pcl::PointXYZRGB>(cloudNow.keypoints,keyPoint_ant_transformed,*cor_all,1, corresname);
    }

    //----------- TEMPLATE CORRESPONDENCIAS -------------//

    template<typename Type, int descriptor> 
    PtrCorrespondences matching(void){

		registration::CorrespondenceEstimation<Type,Type> corEstimation; 

		corEstimation.setInputSource (DescriptorByType); 
		corEstimation.setInputTarget (DescriptorByType);
		PtrCorrespondences correspondences(new pcl::Correspondences); 

		corEstimation.determineReciprocalCorrespondences (*correspondences);
	    //PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", correspondences->size()); 
	    return correspondences;
    }


    //-------------- RANSAC -------------//

    void applyRANSAC(PtrCorrespondences &cor_all_ptr, CloudContainer &cloudNow, CloudContainer &cloudLast,const int &indexMethod){


	    boost::shared_ptr<pcl::Correspondences> cor_inliers(new pcl::Correspondences);
	    
	    PCL_INFO ("Correspondence Rejection Features\n");
		pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> sac;
		sac.setInputSource(cloudNow.keypoints);
		sac.setInputTarget (cloudLast.keypoints);
		sac.setInlierThreshold (0.2);
		sac.setMaximumIterations (1000);
		sac.setInputCorrespondences (cor_all_ptr);
		sac.getCorrespondences (*cor_inliers);

		vResults[indexMethod].badMatches += cor_all_ptr->size()-cor_inliers->size();
		vResults[indexMethod].goodMatches += cor_inliers->size();

		//PCL_INFO ("CORRESPONDENCIAS BUENAS %d ELIMINADAS %d \n ",cor_inliers->size(),cor_all_ptr->size()-cor_inliers->size());


		/*
		Eigen::Matrix4f transformation;
		PtrPCloudRGB cloud_tmp (new PCloudRGB);
		transformation = sac.getBestTransformation();
		   
		pcl::transformPointCloud (*cloudNow.cloud, *cloud_tmp, transformation);
		*/

		Eigen::Matrix4f transformation_tmp;
        Eigen::Matrix4f tranfTemporal ;
        tranfTemporal.setIdentity();
        Eigen::Matrix4f transfomationAnt;
    	PtrPCloudRGB cloud_tmp (new PCloudRGB);
        PtrPCloudRGB cloud_Transformada( new PCloudRGB);
       
        transformation_tmp = sac.getBestTransformation();

        transfomationAnt = transformation;

        tranfTemporal=transformation * transformation_tmp;

        Eigen::Affine3f rotacionAnterior, rotacionActual;
        rotacionAnterior= transfomationAnt;
        rotacionActual= tranfTemporal;

        double distanciaAngular= getRotationDistance(rotacionAnterior,rotacionActual);
        PCL_INFO("DISTANCIA ENTRE NUBES %f \n",distanciaAngular);
        if(distanciaAngular<0.3)
        {
                PCL_INFO("APLICA TRANSFORMACION\n");
                //// MUESTRA LA UNION ENTRE DOS NUBES ///
                transformation = tranfTemporal; 

                //escribir en fichero la transformación estimada
				Eigen::Affine3f transTotal;
				transTotal = transformation;
				pcl::PointXYZRGB p0; //point at zero reference
				p0.x=0; p0.y=0; p0.z=0; p0.r=255; p0.g=0; p0.b=0;
				pcl::PointXYZRGB pt_trans=pcl::transformPoint<pcl::PointXYZRGB>(p0,transTotal); //estimated position of the camera
				Eigen::Quaternion<float> rot2D( (transTotal).rotation());
				myfile <<""<<depthmsg->header.stamp<<" "<<pt_trans.x<<" "<<pt_trans.y<<" "<<pt_trans.z<<" "<<rot2D.x()<<" "<<rot2D.y()<<" "<<rot2D.z()<<" "<<rot2D.w()<<std::endl;


                pcl::transformPointCloud (*cloudNow.cloud, *cloud_tmp, transformation_tmp);
                /// FIN MUESTRA ///
               
                // PCL_INFO("INFORMACION DE ANGULOS %f",angulos);
                pcl::transformPointCloud(*cloudNow.cloud,*cloud_Transformada, transformation);
                cloudNow.tranformCloud =cloud_Transformada;                
               
               generateCloud();
        }

	}


	// ------------ COMBINACIONES DE METODOS ----------//

	void K_NARF_F_NARF(string &resultadosTemp, const int &indexMethod){	
		RangeImage range_image;
		getRangeImage(range_image, cloudNow.cloud);

		pcl::PointCloud<int> keypoint_indicesNARF = applyNARF_Key(cloudNow, range_image);
		removeNaN(cloudNow.keypoints);
		applyNARF_DESC(keypoint_indicesNARF, range_image,cloudNow);
		
		useMethod("K_NARF",F_NARF, indexMethod, resultadosTemp);
	}
	void K_SIFT_F_FPFH(string &resultadosTemp, const int &indexMethod){
		applySIFT(cloudNow);
		removeNaN(cloudNow.keypoints);
		applyFPFH(cloudNow);
		
		useMethod("K_SIFT",F_FPFH, indexMethod, resultadosTemp);
	}
	void K_HARRYS_F_FPFH(string &resultadosTemp, const int &indexMethod){
		applyHarris(cloudNow);
		removeNaN(cloudNow.keypoints);
		applyFPFH(cloudNow);
		
		useMethod("K_HARRYS",F_FPFH, indexMethod, resultadosTemp);
	}
	void K_ISS_F_FPFH(string &resultadosTemp, const int &indexMethod){
		applyISS(cloudNow);		
		removeNaN(cloudNow.keypoints);
		applyFPFH(cloudNow);
	
		useMethod("K_ISS",F_FPFH, indexMethod, resultadosTemp);

	}
	void K_SIFT_F_SHOT(string &resultadosTemp, const int &indexMethod){
		applySIFT(cloudNow);
		removeNaN(cloudNow.keypoints);
		applySHOT(cloudNow);
		
		useMethod("K_SIFT",F_SHOT, indexMethod, resultadosTemp);
	}
	void K_HARRYS_F_SHOT(string &resultadosTemp, const int &indexMethod){
		applyHarris(cloudNow);
		removeNaN(cloudNow.keypoints);
		applySHOT(cloudNow);
		
		useMethod("K_HARRYS",F_SHOT, indexMethod, resultadosTemp);
	}
	void K_ISS_F_SHOT(string &resultadosTemp, const int &indexMethod){
		applyISS(cloudNow);		
		removeNaN(cloudNow.keypoints);
		applySHOT(cloudNow);
		
		useMethod("K_ISS",F_SHOT, indexMethod, resultadosTemp);

	}

	void K_SIFT_F_PFH(string &resultadosTemp, const int &indexMethod){
		applySIFT(cloudNow);		
		removeNaN(cloudNow.keypoints);
		applyPFH(cloudNow);
		
		useMethod("K_SIFT",F_PFH, indexMethod, resultadosTemp);

	}
	void K_HARRYS_F_PFH(string &resultadosTemp, const int &indexMethod){
		applyHarris(cloudNow);		
		removeNaN(cloudNow.keypoints);
		applyPFH(cloudNow);
		
		useMethod("K_HARRYS",F_PFH, indexMethod, resultadosTemp);

	}
	void K_ISS_F_PFH(string &resultadosTemp, const int &indexMethod){		
		applyISS(cloudNow);		
		removeNaN(cloudNow.keypoints);
		applyPFH(cloudNow);
		
		useMethod("K_ISS",F_PFH, indexMethod, resultadosTemp);

	}

	// ------------ FIN COMBINACIONES DE METODOS ----------//

	void useMethod(const string &detector,const int &descriptor,const int &indexMethod, string &resultadosTemp){

		if(iterProcess!=0){

			rutaimagen = "/home/javi/ImagenesPanorama/";
			string sdescriptor="";
			PtrCorrespondences correspondences;

			//------ APLICAR MATCHING -----//

			switch (descriptor){
		 	 	case F_FPFH:  
		  			correspondences = matching<pcl::FPFHSignature33, F_FPFH>();
		  			sdescriptor="F_FPFH";
		  			break;
	  			case F_NARF:
	  				correspondences = matching<pcl::Narf36, F_NARF>();
	  				sdescriptor = "F_NARF";
	  				break;
	  			case F_SHOT:
	  				correspondences = matching<pcl::SHOT352, F_SHOT>();
	  				sdescriptor = "F_SHOT";
	  				break;
	  			case F_PFH:
	  				correspondences = matching<pcl::PFHSignature125, F_PFH>();
	  				sdescriptor = "F_PFH";
			} 
						
			cout<<"APLICO RANSAC"<<endl;	
			//----- APLICAR RANSAC ----//
			applyRANSAC(correspondences, cloudNow, cloudLast, indexMethod);
			cout<<"FIN RANSAC con "<<detector<<"+"<<sdescriptor<<endl;

			viewCorrespondences(correspondences);
			stringstream ssNumber;
		    ssNumber << iterProcess;
		    rutaimagen+= ssNumber.str();
		    rutaimagen+=detector+"+"+sdescriptor;
			viewer->saveScreenshot (rutaimagen);
			
			
            //------ RESULTADOS -----//

		    clock_gettime( CLOCK_REALTIME, &ts2 ); 
		    vResults[indexMethod].tiempo +=(float) ( 1.0*(1.0*ts2.tv_nsec - ts1.tv_nsec*1.0)*1e-9+ 1.0*ts2.tv_sec - 1.0*ts1.tv_sec );;
			resultadosTemp += vResults[indexMethod].print(detector, sdescriptor);
		}
	}

	//--------GENERAR MAPA---------//
    double getRotationDistance(const Eigen::Affine3f &matrizAnt, const Eigen::Affine3f &matrizActu){

        float rollAnt,pitchAnt,yawAnt;
        float rollAct,pitchAct,yawAct;
       
        pcl::getEulerAngles (matrizAnt, rollAnt, pitchAnt, yawAnt);
        pcl::getEulerAngles (matrizActu,rollAct, pitchAct, yawAct);
       
        PCL_INFO("Matriz anteriro roll: %f, pitch: %f, yaw: %f\n",rollAnt,pitchAnt,yawAnt);
        PCL_INFO("Matriz actual roll: %f, pitch: %f, yaw: %f\n",rollAct,pitchAct,yawAct);

        return sqrt((rollAct-rollAnt)*(rollAct-rollAnt) + (pitchAct-pitchAnt)*(pitchAct-pitchAnt) + (yawAct-yawAct)*(yawAct-yawAct));
    }

	void generateCloud(){
	    PtrPCloudRGB cloud_trans (new PCloudRGB);
	    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
	    *cloud_trans = *cloudNow.tranformCloud;


	     for(int i=0;i<cloudNow.cloud->size();i++)
	                mergedCloud->push_back(cloud_trans->at(i));

         float voxel_side_size=0.02f; // me quedaria con un representante para cada 2cm cúbicos de la nube

        pcl::VoxelGrid <pcl::PointXYZRGB> sor;
        sor.setInputCloud (mergedCloud);    //la nube mapa necesita ser filtrada
        sor.setLeafSize (voxel_side_size, voxel_side_size, voxel_side_size);
        sor.filter (*cloud_filtered);    //la filtramos en otra nube auxiliar

        mergedCloud->clear();            //borramos la nube mapa anterior
        std::swap(mergedCloud,cloud_filtered);    //e intercambiamos los punteros pa


            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud(mergedCloud);
        if (!viewerTransform->updatePointCloud (mergedCloud,rgbcloud, "cloudn4")) //intento actualizar la nube y si no existe la creo.
            viewerTransform->addPointCloud(mergedCloud,rgbcloud,"cloudn4");

    }
	
	void processRegistration(){

		//Limpiamos para poder visualizar los Keypoints
		viewer->removeAllPointClouds();

		const float* depthImageFloat = reinterpret_cast<const float*>(&depthmsg->data[0]);

		cloudNow.cloud = getCloudColorDepth(imageColormsg->image, depthImageFloat);
		
		removeNaN(cloudNow.cloud);

		//****** VISUALIZACION DE LA NUBE ******//

		//Manejador de color de la nube "cloud"
		visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloudNow.cloud);   

		//Actualiza la nube y si no existe la creo.
		if (!viewer->updatePointCloud (cloudNow.cloud,rgb, "cloud")){
		    viewer->addPointCloud(cloudNow.cloud,rgb,"cloud");
		}

		//----- APLICAR METODOS -----//
		
		string resultadosTemp = "";

		for(short i=0; i<nMethods;++i){
			clock_gettime( CLOCK_REALTIME, &ts1 );
			(this->*vMethods[i])(resultadosTemp, i);
		}
		
		cout<<"VOLCAR RESULTADOS FINAL"<<endl;
		//----- VOLCAR RESULTADOS -----//
		RESULTADOSFIN = resultadosTemp;

		cloudLast = cloudNow;
		++iterProcess;

		cout<< "ITERACIONES "<<iterProcess<<endl;
	}



	
	//Procesamos ImagenProfundidad y activamos su recepcion
	void imageCbdepth(const sensor_msgs::ImageConstPtr& msg){

	    depthmsg = msg;
	    depthreceived=true;
	    if(imagereceived && depthreceived)
	        processRegistration();

	}
	//Procesamos ImagenColor y activamos su recepcion
	void imageCb(const sensor_msgs::ImageConstPtr& msg){
	    try
	    {
	      imageColormsg = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	    }
	    catch (cv_bridge::Exception& e)
	    {
	      ROS_ERROR("cv_bridge exception: %s", e.what());
	      return;
	    }
	    //std::cerr<<" imagecb: "<<msg->header.frame_id<<" : "<<msg->header.seq<<" : "<<msg->header.stamp<<std::endl;
	    imagereceived=true;
	    if(imagereceived && depthreceived)
	        processRegistration();
	}
	void bucle_eventos(){

	    while (ros::ok()){
	      ros::spinOnce();       // Handle ROS events
	      //cloud_viewer_regi.spinOnce(1);  //update viewers
	      //cloud_viewer.spinOnce(1);
	      viewer->spinOnce(1);
	    }
	}
	
/** Cerrar fichero, lo llamaremos cuando recibamos la señal de corte de ejecución Ctrl+C, escribe los resultados, cierra el fichero y sale del programa  **/
static void cierraFichero (int param){	

	clock_gettime( CLOCK_REALTIME, &tEnd );
	float tiempoTotal =(float) ( 1.0*(1.0*tEnd.tv_nsec - tStart.tv_nsec*1.0)*1e-9+ 1.0*tEnd.tv_sec - 1.0*tStart.tv_sec );;
	FICHERO << "Tiempo total de ejecucion: " << tiempoTotal<<" s"<<endl<<endl;
	FICHERO << RESULTADOSFIN;
	FICHERO.close();
  	exit(0);
  	kill(0, SIGKILL);
  }

};


int main(int argc, char **argv){

	clock_gettime( CLOCK_REALTIME, &tStart );

	ros::init (argc, argv, "pub_pcl");
	Panorama3D p;

	p.bucle_eventos();

	cout<<"CIERRO TRAS ERROR"<<endl;
	Panorama3D::cierraFichero(0);
	
	return 0; 
}
