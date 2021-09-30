
<?php>





// ini_set("display_errors", "1");





$uploaddir = '/var/www/html/rscripts/rds/';
$uploadfile = $uploaddir . basename($_FILES['imgfile']['name']);

// echo "파일 업로드 경로: ";
// var_dump($uploaddir);
// echo '<br />';
// echo "원본 파일 이름: ";
// var_dump(basename($_FILES['imgfile']['name']));

// echo '<pre>';



// function dirsize($dir){ 
	   // static $size, $cnt; 
	   // $fp = opendir($dir); 
	   // while(false !== ($entry = readdir($fp))){ 
			 // if(($entry != ".") && ($entry != "..")){ 
				  // if(is_dir($dir.'/'.$entry)){ 
					   // clearstatcache(); 
					   // dirsize($dir.'/'.$entry); 
				  // } else if(is_file($dir.'/'.$entry)){ 
					   // $size += filesize($dir.'/'.$entry); 
					   // clearstatcache(); 
					   // $cnt++; 
				  // } 
			 // } 
	   // } 
      // closedir($fp); 

      // $stat = array( 
                // 'size' => $size, 
                // 'cnt' => $cnt 
      // ); 
      // return $stat; 
 // }

// function attach($size) { 
      // if($size < 1024){ 
            // return number_format($size*1.024).'b'; 
      // } else if(($size > 1024) && ($size < 1024000)){ 
            // return number_format($size*0.001024).'Kb'; 
      // } else if($size > 1024000){ 
            // return number_format($size*0.000001024,2).'Mb'; 
      // } 
      // return 0; 
 // }



// $arr = dirsize('/var/www/html/rscripts/'); 
// echo "총 파일수: ".$arr['cnt']." 총 파일 용량: ".attach($arr['size']); 
// echo .$arr['cnt']; 


$dir = '/var/www/html/rscripts/rds/';

// $dirsize = `du -sh $dir | awk '{print $1}'`;
$dirsize = `du -s $dir`;

// if ($dirsize>=12817952)
	// echo "yeah";
// else
	// echo "TT";
// echo $dirsize;


if ($dirsize<100000000)
	move_uploaded_file($_FILES['imgfile']['tmp_name'], $uploadfile)

// if (move_uploaded_file($_FILES['imgfile']['tmp_name'], $uploadfile))
// {
	// echo "파일이 유효하고, 성공적으로 업로드 되었습니다. \n";
// }
// else
// {
	// print "파일 업로드 공격의 가능성이 있습니다. \n";
// }
// echo '자세한 디버깅 정보입니다 :';
// print_r($_FILES);
// print "</pre>";


	
?>


