/*A continuación os vamos a mostrar como transformar una nube de puntos, como añadirla a nuestro mapa y por ultimo como reducir el mapa con la función VoxelGrid de pcl:
*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*cloud, *cloud_transformed,transFinal);


    for(int i=0;i<cloud->size();i++){
        mapa->push_back(cloud_transformed->at(i));
    }

    sensor_msgs::PointCloud2::Ptr mapamsg(new sensor_msgs::PointCloud2),mapamsgfiltered(new sensor_msgs::PointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::toROSMsg(*mapa,*mapamsg);

    pcl::VoxelGrid <sensor_msgs::PointCloud2 > sor;
    sor.setInputCloud (mapamsg);
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    sor.filter (*mapamsgfiltered);

    pcl::fromROSMsg(*mapamsgfiltered,*mapa);

 
/*
primero transformamos la nube de puntos actual al sistema de coordenadas de nuestro mapa de la misma forma que hicimos para medir el error de la transformación. Luego añadimos esos puntos a nuestro mapa que no es mas que una nube de puntos de pcl (pcl::PointCloud<pcl::PointXYZ>::Ptr). Por ultimo, le aplicamos el filtro VoxelGrid con el fin de reducir el numero de puntos de nuestro mapa, en este caso solo nos quedariamos con un punto representante en cada cubo virtual de 1 cm de lado.
*/