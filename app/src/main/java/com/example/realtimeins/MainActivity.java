package com.example.realtimeins;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.numbers.quaternion.Quaternion;
import org.apache.commons.numbers.quaternion.Slerp;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class MainActivity extends AppCompatActivity {

    private static final double dt = 1.0d / 200.0d;
    private static final int stepSize = 50;
    private static final double stepTime = dt * stepSize;
    private static final int windowSize = 200;
    private static final double NS2S = 1.0d / 1000000000.0d;
    private static final int featuresDim = 6;
    double startTimestamp = 0.0d;
    double stepCount = 0.0d;
    private static double delayTime = 0;
    private static boolean isAllSensorStart = false;
    float xx = 0;
    float yy = 0;
    public static final int width = 400;
    public static final int height = 400;
    public static final float StrokeWidth = 2.0f;

    private SensorManager mSensorManager = null;

    public TextView x, y, t, st, location_x, location_y;
    ImageView cood;

    List<double[]> gyroList = new ArrayList<>();

    List<Quaternion> grvList = new ArrayList<>();
    List<Double> grvTList = new ArrayList<>();

    List<Double> acceXList = new ArrayList<>();
    List<Double> acceYList = new ArrayList<>();
    List<Double> acceZList = new ArrayList<>();
    List<Double> acceTList = new ArrayList<>();

    List<Double> gyroUncalibXList = new ArrayList<>();
    List<Double> gyroUncalibYList = new ArrayList<>();
    List<Double> gyroUncalibZList = new ArrayList<>();
    List<Double> gyroUncalibTList = new ArrayList<>();
    float[][] features = new float[windowSize][featuresDim];

    double[] bias = null;
    double[] outputTime = new double[windowSize];

    LinearInterpolator li = new LinearInterpolator();

    public static Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    public static Canvas canvas = new Canvas(bitmap);
    public static Paint paint = new Paint();

    private static SensorEventListener listener;
    Sensor mGyroSensor, mAcceSensor, mGameRV, mGyroUncalib;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        x = findViewById(R.id.x);
        y = findViewById(R.id.y);
        location_x = findViewById(R.id.location_x);
        location_y = findViewById(R.id.location_y);
        t = findViewById(R.id.time);
        st = findViewById(R.id.stepTime);
        cood = findViewById(R.id.cood);

        canvas.drawColor(Color.BLACK);
        cood.setImageBitmap(bitmap);
        paint.setColor(Color.WHITE);
        paint.setStrokeWidth(StrokeWidth);

        st.setText(Double.toString(stepTime));

        listener = new Listener();
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        mAcceSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mGyroSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mGameRV = mSensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
        mGyroUncalib = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED);


    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(listener, mGyroSensor, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(listener, mAcceSensor, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(listener, mGameRV, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(listener, mGyroUncalib, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener((SensorEventListener) this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mSensorManager.unregisterListener((SensorEventListener) this);
    }

    class Listener implements SensorEventListener {

        @RequiresApi(api = Build.VERSION_CODES.N)
        @Override
        public void onSensorChanged(SensorEvent event) {
            Sensor sensor = event.sensor;

            if(startTimestamp==0) {
                startTimestamp = event.timestamp * NS2S;
            }

            if(sensor.getType() == Sensor.TYPE_GYROSCOPE && gyroList.size() == 0) {
                float gyroX = event.values[0];
                float gyroY = event.values[1];
                float gyroZ = event.values[2];
                gyroList.add(new double[]{gyroX, gyroY, gyroZ});
            }
            if(sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
                acceXList.add((double) event.values[0]);
                acceYList.add((double) event.values[1]);
                acceZList.add((double) event.values[2]);
                acceTList.add(((double) event.timestamp * NS2S) - startTimestamp);
            }
            if(sensor.getType() == Sensor.TYPE_GAME_ROTATION_VECTOR) {
                grvList.add(Quaternion.of(event.values[3], event.values[0], event.values[1], event.values[2]));
                grvTList.add(((double) event.timestamp * NS2S) - startTimestamp);
            }
            if(sensor.getType() == Sensor.TYPE_GYROSCOPE_UNCALIBRATED && gyroList.size() > 0) {
                double[] gyroUncalib = {event.values[0], event.values[1], event.values[2]};
                if(bias==null) {
                    sumBias(gyroUncalib, (double[]) gyroList.get(0));
                    bias = gyroUncalib;
                } else {
                    sumBias(gyroUncalib, bias);
                }
                gyroUncalibXList.add(gyroUncalib[0]);
                gyroUncalibYList.add(gyroUncalib[1]);
                gyroUncalibZList.add(gyroUncalib[2]);
                gyroUncalibTList.add(((double) event.timestamp * NS2S) - startTimestamp);
            }
            if(!isAllSensorStart) {
                if(gyroUncalibTList.size()>1&&grvTList.size()>1&&acceTList.size()>1) {
                    delayTime = event.timestamp*NS2S - startTimestamp;
                    for(int i=0;i<outputTime.length;i++) outputTime[i] = (dt * i) + delayTime;
                    isAllSensorStart = true;
                }
            }

            if((event.timestamp*NS2S) - startTimestamp > (dt*windowSize) + (stepCount*stepTime) + delayTime) {
                double endTime = (dt*windowSize) + (stepCount*stepTime) + delayTime;
                if(acceTList.get(acceTList.size()-1) > endTime &&
                        gyroUncalibTList.get(gyroUncalibTList.size()-1) > endTime &&
                        grvTList.get(grvTList.size()-1) > endTime) {
                    if(stepCount>0) {
                        for(int i=0;i<outputTime.length;i++) {
                            outputTime[i] += stepTime;
                        }
                    }

                    // Lerp
                    double[] gyroUncalibXArr = gyroUncalibXList.stream().mapToDouble(Double::doubleValue).toArray();
                    double[] gyroUncalibYArr = gyroUncalibYList.stream().mapToDouble(Double::doubleValue).toArray();
                    double[] gyroUncalibZArr = gyroUncalibZList.stream().mapToDouble(Double::doubleValue).toArray();
                    double[] gyroUncalibInputTime = gyroUncalibTList.stream().mapToDouble(Double::doubleValue).toArray();
                    gyroUncalibXList = linearInterpolate(gyroUncalibXArr, gyroUncalibInputTime, outputTime);
                    gyroUncalibYList = linearInterpolate(gyroUncalibYArr, gyroUncalibInputTime, outputTime);
                    gyroUncalibZList = linearInterpolate(gyroUncalibZArr, gyroUncalibInputTime, outputTime);
                    gyroUncalibTList = DoubleStream.of(outputTime).boxed().collect(Collectors.toList());

                    // Lerp
                    double[] acceXArr = acceXList.stream().mapToDouble(Double::doubleValue).toArray();
                    double[] acceYArr = acceYList.stream().mapToDouble(Double::doubleValue).toArray();
                    double[] acceZArr = acceZList.stream().mapToDouble(Double::doubleValue).toArray();
                    double[] acceInputTime = acceTList.stream().mapToDouble(Double::doubleValue).toArray();
                    acceXList = linearInterpolate(acceXArr, acceInputTime, outputTime);
                    acceYList = linearInterpolate(acceYArr, acceInputTime, outputTime);
                    acceZList = linearInterpolate(acceZArr, acceInputTime, outputTime);
                    acceTList = DoubleStream.of(outputTime).boxed().collect(Collectors.toList());

                    // Slerp
                    int outputTimeCount = 0;
                    int grvTimeCount = 0;
                    int i = 0;
                    List<Quaternion> newGrvList = new ArrayList<>();
                    if(stepCount>0) {
                        outputTimeCount = windowSize-stepSize;
                        grvTimeCount = windowSize-1;
                        i = windowSize-stepSize;
                        for(int j=stepSize;j<windowSize;j++) {
                            newGrvList.add(grvList.get(j));
                            t.setText(Double.toString(grvTList.get(200)-grvTList.get(199)));
                        }
                    }
                    int[] left = new int[windowSize];
                    int[] right = new int[windowSize];
                    double leftTime, rightTime;
                    double[] slerpTime = new double[windowSize];
                    while(outputTimeCount<windowSize) {
                        while (grvTList.get(grvTimeCount) >= outputTime[outputTimeCount]) {
                            left[outputTimeCount] = grvTimeCount - 1;
                            leftTime = grvTList.get(grvTimeCount - 1);
                            right[outputTimeCount] = grvTimeCount;
                            rightTime = grvTList.get(grvTimeCount);
                            slerpTime[outputTimeCount] = (outputTime[outputTimeCount] - leftTime) / (rightTime - leftTime);
                            outputTimeCount++;
                            if (outputTimeCount == windowSize) break;
                        }
                        grvTimeCount++;
                    }
                    while(i<windowSize) {
                        newGrvList.add(new Slerp(grvList.get(left[i]), grvList.get(right[i])).apply(slerpTime[i]));
                        i++;
                    }
                    grvList = newGrvList;
                    grvTList = DoubleStream.of(outputTime).boxed().collect(Collectors.toList());

                    // 입력값으로 사용할 features 만들기
                    float[][] newFeatures = new float[windowSize][featuresDim];
                    i = 0;
                    if(stepCount>0) {
                        i = windowSize-stepSize;
                        System.arraycopy(features, stepSize, newFeatures, 0, windowSize-stepSize);
                    }
                    while(i<windowSize) {
                        Quaternion qAcce = Quaternion.of(0.0d, acceXList.get(i), acceYList.get(i), acceZList.get(i));
                        Quaternion qGyroUncalib = Quaternion.of(0.0d, gyroUncalibXList.get(i), gyroUncalibYList.get(i), gyroUncalibZList.get(i));
                        Quaternion globAcce = grvList.get(i).multiply(qAcce).multiply(grvList.get(i).conjugate());
                        Quaternion globGyroUncalib = grvList.get(i).multiply(qGyroUncalib).multiply(grvList.get(i).conjugate());
                        double[] acceArr = globAcce.getVectorPart();
                        double[] gyroUncalibArr = globGyroUncalib.getVectorPart();
                        newFeatures[i] = new float[]{(float)gyroUncalibArr[0], (float)gyroUncalibArr[1], (float)gyroUncalibArr[2],
                                (float)acceArr[0], (float)acceArr[1], (float)acceArr[2]};
                        i++;
                    }
                    features = newFeatures;

                    // 전치행렬 만들기
                    float[][][] featuresT = new float[1][featuresDim][windowSize];
                    for(int j=0;j<windowSize;j++) {
                        featuresT[0][0][j] = features[j][0];
                        featuresT[0][1][j] = features[j][1];
                        featuresT[0][2][j] = features[j][2];
                        featuresT[0][3][j] = features[j][3];
                        featuresT[0][4][j] = features[j][4];
                        featuresT[0][5][j] = features[j][5];
                    }

                    float[][] output = new float[1][2];
                    try (Interpreter interpreter = getTfliteInterpreter("my_model.tflite")) {
                        assert interpreter != null;
                        interpreter.run(featuresT, output);
                    }

                    xx += output[0][0]*stepTime;
                    yy += output[0][1]*stepTime;
                    canvas.drawLine(
                            Float.parseFloat((String) location_x.getText())*10+((float) width/2),
                            Float.parseFloat((String) location_y.getText())*10+((float) height/2),
                            xx*10+((float) width/2),
                            yy*10+((float) height/2),
                            paint);
                    location_x.setText(String.valueOf(xx));
                    location_y.setText(String.valueOf(yy));
                    x.setText(String.valueOf(output[0][0]));
                    y.setText(String.valueOf(output[0][1]));

                    stepCount += 1;
                }
            }
        }

        private Interpreter getTfliteInterpreter(String modelPath) {
            try {
                return new Interpreter(loadModelFile(MainActivity.this, modelPath));
            }
            catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }

        // 모델을 읽어오는 함수로, 텐서플로 라이트 홈페이지에 있다.
        // MappedByteBuffer 바이트 버퍼를 Interpreter 객체에 전달하면 모델 해석을 할 수 있다.
        private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
            AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }

        @RequiresApi(api = Build.VERSION_CODES.N)
        List<Double> linearInterpolate(double[] value, double[] in_t, double[] out_t) {
            double[] newValue = new double[windowSize];
            int i = 0;
            if(stepCount>0) {
                System.arraycopy(value, stepSize, newValue, 0, windowSize-stepSize);
                i = windowSize-stepSize;
            }

            PolynomialSplineFunction psf = li.interpolate(in_t, value);
            while(i<windowSize) {
                newValue[i] = psf.value(out_t[i]);
                i++;
            }

            return DoubleStream.of(newValue).boxed().collect(Collectors.toList());
        }

        private void sumBias(double[] gyroUncalib, double[] gyro) {
            for(int i=0;i<gyroUncalib.length;i++) {
                gyroUncalib[i] -= gyro[i];
            }
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int i) {
        }
    }
}