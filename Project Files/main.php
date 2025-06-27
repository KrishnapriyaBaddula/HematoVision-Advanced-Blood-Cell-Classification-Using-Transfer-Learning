// PHP sends image to Flask API
$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, "http://localhost:5000/predict");
curl_setopt($ch, CURLOPT_POSTFIELDS, ['image' => new CURLFile('test.jpg')]);
$response = curl_exec($ch);
curl_close($ch);
