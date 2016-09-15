#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <string> 
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

using namespace cv;
using namespace std;
using namespace cv_bridge;

static const std::string OPENCV_WINDOW = "Image window";

namespace enc = sensor_msgs::image_encodings;

//Globales que se ocupan de escribir los resultados en el fichero de salida
//muy a mi pesar es la unica manera de evitar lidiar con los problemas de variables
//y metodos estaticos al usar punteros a funcion
ofstream FICHERO;
string RESULTADOSFIN;
//tiempos inicio y fin de programa
struct timespec tStart, tEnd;

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

class ImageContainer{
		
		public:

		//Caracteristicas segun el metodo
		//Vector que se corresponde del modo indice = metodo
		//Y en cada metodo se asocia un par keypoints, descriptor
		vector< pair< vector<KeyPoint>,Mat > > vMethodKeyDescriptor;

		Mat image;

		ImageContainer & operator=(const ImageContainer &i){
			if(&i==this)
				return *this;
			else{
				vMethodKeyDescriptor.clear();
				vMethodKeyDescriptor = i.vMethodKeyDescriptor;
				image = i.image;	
			}
			return *this;
		}
};

class Panorama{

private:

	// Necesario para it_
	ros::NodeHandle nh_;
	//Manejo de imagenes ROS
	image_transport::ImageTransport it_;
	//Subscriptor ROS
	image_transport::Subscriber image_sub_;


	//La imagen que analizamos previamente
	ImageContainer imageLast;
	//La imagen que estamos analizando en la iteracion actual
	ImageContainer imageNow;
	//Numero de imagenes analizadas
	int numberImage;
	//La ruta donde guardamos la imagen del match
	string rutaimagen;

	//Necesitamos saber si nos encontramos en la primera iteracion para no comparar con la imagen anterior, la cual seria null
	bool primeraIteracion;

	//Vector de metodos pares <detector, descriptor>
	vector< pair< string,string > > vMethods;
	int nMethods;

	//Umbral establecido para restringir el matching
	int umbral;

	//Vector de resultados para cada metodo
	vector< Results > vResults;

	//Estructura para medir tiempos del algoritmo
	struct timespec ts1, ts2; 

public:


	Panorama() : it_(nh_){
	// Cuando llegue una imagen al topic, llama a la funcion imgeCB
		image_sub_ = it_.subscribe("/camera/rgb/image_color", 1, &Panorama::imageCb, this);

		//PcCAM Topic
		//image_sub_ = it_.subscribe("/camera/image_raw", 1, &Panorama::imageCb, this);

		//	vMethods.push_back(make_pair("SIFT","SIFT"));
		//vMethods.push_back(make_pair("SIFT","BRIEF"));
		//vMethods.push_back(make_pair("SIFT","SURF"));
		/*vMethods.push_back(make_pair("SIFT","ORB"));*/
		//vMethods.push_back(make_pair("SURF","SIFT"));
		//vMethods.push_back(make_pair("SURF","SURF"));
		//vMethods.push_back(make_pair("SURF","BRIEF"));
		/*vMethods.push_back(make_pair("SURF","ORB"));*/
		//vMethods.push_back(make_pair("FAST","SURF"));
		//vMethods.push_back(make_pair("FAST","SIFT"));
		//vMethods.push_back(make_pair("FAST","BRIEF"));
		//vMethods.push_back(make_pair("ORB","SIFT"));
		//vMethods.push_back(make_pair("ORB","SURF"));
		vMethods.push_back(make_pair("ORB","ORB"));
		//vMethods.push_back(make_pair("MSER","SIFT"));
		//vMethods.push_back(make_pair("MSER","SURF"));
		//vMethods.push_back(make_pair("MSER","ORB"));
		//vMethods.push_back(make_pair("MSER","BRIEF"));

		nMethods = vMethods.size();

		//Reservamos espacio en el vector de resultados, tanto como metodos haya
		vResults.resize(nMethods);

		//Controlamos la señal de corte de ejecucion para cerrar el fichero de resultados
		void (*handler)(int);
		handler = signal(SIGINT, cierraFichero);

  		FICHERO.open ("/home/javi/resultados.txt");
  		RESULTADOSFIN="";

  		numberImage = 0;
  		primeraIteracion = true;
  		umbral = 200;
	}

void imageCb(const sensor_msgs::ImageConstPtr& msg){

	CvImagePtr cv_ptr = imageRosToCvGrey(msg);

    //Pasamos la imagen a escala de grises y color a tipo MAT
    Mat src_gray, imageColor;	    
	cvtColor( cv_ptr->image, src_gray, CV_BGR2GRAY );
    cvtColor( src_gray, imageColor, CV_GRAY2BGR );
    if(src_gray.rows==0){return;} 

    /* IMAGEN ACTUAL */
	imageNow.image = imageColor;

	string mDetector;
	string mDescriptor;

	string resultadosTemp = "";

	for(int i=0; i<nMethods; ++i){
		clock_gettime( CLOCK_REALTIME, &ts1 );


		mDetector = vMethods[i].first;
		mDescriptor = vMethods[i].second;

		rutaimagen = "/home/javi/ImagenesPanorama/";

		applyMethod(mDetector,mDescriptor, src_gray, imageNow);

		if(!primeraIteracion){

			/*MATCHING*/

			vector< DMatch > good_matches;
			Mat img_matches;
			//matchingFlann(imageNow, good_matches, img_matches, i);
			matchingKnn(imageNow, good_matches, img_matches, i);

			/**PINTAMOS Keypoints y Matches**/
			vector<KeyPoint> keypointsNow = imageNow.vMethodKeyDescriptor[i].first;
			vector<KeyPoint> keypointsLast = imageLast.vMethodKeyDescriptor[i].first;

			drawMatches(imageNow.image, keypointsNow, imageLast.image, keypointsLast, good_matches, img_matches);
		    imshow("match", img_matches);

		    stringstream ssNumber;
		    ssNumber << numberImage;
		    rutaimagen+= ssNumber.str();
		    rutaimagen+=mDetector+"+"+mDescriptor+".jpg";
		    cout<<rutaimagen<<endl;
		  	imwrite( rutaimagen, img_matches );

		    /* APLICAMOS RANSAC */

	 		vector<Point2f> obj;
			vector<Point2f> scene;
			Mat H;
			for( int j = 0; j < good_matches.size(); ++j )
			{
				//-- Get the keypoints from the good matches
				obj.push_back(keypointsNow[ good_matches[j].queryIdx ].pt );
				scene.push_back(keypointsLast  [ good_matches[j].trainIdx ].pt );
			}
	
			if(obj.size()>4 && scene.size()>4){
				Mat H = findHomography( obj, scene, CV_RANSAC );
				ransac(keypointsNow, keypointsLast, H, good_matches, src_gray, i);
			}

		    /* GUARDAR DATOS */
		    clock_gettime( CLOCK_REALTIME, &ts2 ); 
		    vResults[i].tiempo +=(float) ( 1.0*(1.0*ts2.tv_nsec - ts1.tv_nsec*1.0)*1e-9+ 1.0*ts2.tv_sec - 1.0*ts1.tv_sec );;
			resultadosTemp += vResults[i].print(mDetector, mDescriptor);
		}
	}
	RESULTADOSFIN = resultadosTemp;
	//cout << resultadosTemp;
    imageLast = imageNow;
    imageNow.vMethodKeyDescriptor.clear();
    numberImage ++;

    primeraIteracion = false;
    cv::waitKey(3);

}
	/* Aplica metodo de estraccion de caracteristicas concreto */
	void applyMethod(string methodDetector,string methodDescriptor, Mat src_gray, ImageContainer &imageNow){

		Mat descriptorFound;
		vector<KeyPoint> keypoints;
		
		Ptr<FeatureDetector> detector = FeatureDetector::create(methodDetector);

		Ptr<DescriptorExtractor> exDescriptor = DescriptorExtractor::create(methodDescriptor);

		detector->detect(src_gray, keypoints);
		exDescriptor->compute(src_gray, keypoints, descriptorFound);

		//asegurar el tipo de dato que devuelven segun el metodo
		if(descriptorFound.type()!=CV_32F) 
			descriptorFound.convertTo(descriptorFound, CV_32F); 
		//Asignamos los keypoints y el descriptor del metodo correspondiente
		imageNow.vMethodKeyDescriptor.push_back(make_pair(keypoints,descriptorFound));
		
	}

	void matchingFlann(ImageContainer &imageNow, vector< DMatch > &good_matches, Mat &img_matches,const int &indexMethod){

		Mat descriptorNow = imageNow.vMethodKeyDescriptor[indexMethod].second;
		Mat descriptorLast = imageLast.vMethodKeyDescriptor[indexMethod].second;

		FlannBasedMatcher matcher;
		vector< DMatch > matches;
		matcher.match( descriptorNow, descriptorLast, matches );
		//double max_dist = 0; 
		double min_dist = 100;


		for( int i = 0; i < descriptorNow.rows; ++i ){ 
			double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			//if( dist > max_dist ) max_dist = dist;
		}

		/**COMENTAR LA MAXIMA DISTANCIA Y LA  MINIMA POR ESO NO USAMOS LA MEDIA*/
//		printf("-- Max dist : %f \n", max_dist );
//		printf("-- Min dist : %f \n", min_dist );

		/* Establecemos un multiplicador para ampliar el umbral basado en la minima distancia y a parte otro arbitrario */ 
		//if( matches[i].distance < 2*min_dist && matches[i].distance<humbralDado )
		for( int i = 0; i < descriptorNow.rows; ++i ){ 
			if( matches[i].distance < 1.5*min_dist && matches[i].distance <umbral){ 
				good_matches.push_back( matches[i]); 
			}
		}
	}
	void matchingKnn(ImageContainer &imageNow, vector< DMatch > &good_matches, Mat &img_matches,const int &indexMethod){

		Mat descriptorNow = imageNow.vMethodKeyDescriptor[indexMethod].second;
		Mat descriptorLast = imageLast.vMethodKeyDescriptor[indexMethod].second;

		FlannBasedMatcher matcher;
		vector< vector< DMatch > > matches;
		//al ser KNN, indicamos 2k vecinos
		matcher.knnMatch( descriptorNow, descriptorLast, matches, 2 );
		
		for( int i = 0; i < matches.size(); ++i ){ 
			if(matches[i].size()>=2){
				//Punteros a los 2 vecinos
				//cogemos los dos vecinos de menor distancia
				const DMatch &m1 = matches[i][0];
				const DMatch &m2 = matches[i][1];
				if(m1.distance <= 0.8 * m2.distance && m1.distance<umbral)
					good_matches.push_back(m1);
			} 

		}
	}
	void ransac(const vector<KeyPoint> &keypoints_ant, const vector<KeyPoint> &currentKeypoints, const Mat &H, const vector<DMatch> &good_matches, const Mat &src_gray, const int &indexMethod)
	{
		const std::vector<Point2f> points_ant_transformed(keypoints_ant.size());
		std::vector<Point2f> keypoints_ant_vector(keypoints_ant.size());
		cv::KeyPoint::convert(keypoints_ant,keypoints_ant_vector);

		//transformamos los puntos de la imagen anterior
		perspectiveTransform( keypoints_ant_vector, points_ant_transformed, H);

		//creamos una copia de la imagen actual que usaremos para dibujar
		Mat transformed_image;
		cvtColor(src_gray, transformed_image, CV_GRAY2BGR);

		//los que esten mas lejos que este parametro se consideran outliers (o que la transformacion está mal calculada)
		//este valor es orientativo, podeis cambiarlo y ajustarlo a los valores
		float distance_threshold=10.0;
		int contdrawbuenos=0;
		int contdrawmalos=0;
		for ( int i =0;i<good_matches.size();i++)
		{
		    int ind        = good_matches.at(i).trainIdx ;
		    int ind_Ant    = good_matches.at(i).queryIdx;

		    cv::Point2f p=        currentKeypoints.at(ind).pt;
		    cv::Point2f p_ant=    points_ant_transformed[ind_Ant];

		    circle( transformed_image, p_ant, 5, Scalar(255,0,0), 2, 8, 0 ); //ant blue
		    circle( transformed_image, p, 5, Scalar(0,255,255), 2, 8, 0 ); //current yellow

		    Point pointdiff = p - points_ant_transformed[ind_Ant];
		        float distance_of_points=cv::sqrt(pointdiff.x*pointdiff.x + pointdiff.y*pointdiff.y);

		    if(distance_of_points < distance_threshold){ // los good matches se pintan con un circulo verde mas grand
		        contdrawbuenos++;
		        circle( transformed_image, p, 9, Scalar(0,255,0), 2, 8, 0 ); //current red
		    }
		    else{
		        contdrawmalos++;
		        line(transformed_image,p,p_ant,Scalar(0, 0, 255),1,CV_AA);
		    }
		}
		//Guardamos los matches buenos y malos correspondientes al metodo
		vResults[indexMethod].badMatches += contdrawmalos;
		vResults[indexMethod].goodMatches += contdrawbuenos;
		imshow( "transformed", transformed_image );
	}


	/* Transforma una imagen tipo ROS a OpenCV */
	CvImagePtr imageRosToCvGrey(const sensor_msgs::ImageConstPtr& msg){

		CvImagePtr cv_ptr;
		try{
		  cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
		}
		catch (cv_bridge::Exception& e){
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return cv_ptr;
		}

		return cv_ptr;
	}
	/** Cerrar fichero, lo llamaremos cuando recibamos la señal de corte de ejecución Ctrl+C, escribe los resultados, cierra el fichero y sale del programa  **/
	static void cierraFichero (int param){	

		clock_gettime( CLOCK_REALTIME, &tEnd );
		float tiempoTotal =(float) ( 1.0*(1.0*tEnd.tv_nsec - tStart.tv_nsec*1.0)*1e-9+ 1.0*tEnd.tv_sec - 1.0*tStart.tv_sec );;
		FICHERO << "Tiempo total de ejecucion: " << tiempoTotal<<" s"<<endl<<endl;
		FICHERO << RESULTADOSFIN;
		FICHERO.close();
	  	cout << "FIN DEL PROGRAMA"<< endl;
	  	exit(0);
	  }
};

int main(int argc, char **argv) {
	
	//Tomamos la medida de tiempo al inicio del programa para ver el tiempo total cuando acabe
	clock_gettime( CLOCK_REALTIME, &tStart );

	cv::initModule_nonfree();
	// Inicializa un nuevo nodo llamado Panorama
	ros::init(argc, argv, "panorama"); 
	Panorama p;

	ros::spin();
	return 0;
}
