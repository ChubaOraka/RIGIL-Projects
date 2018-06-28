package com.rigil.newtest;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.view.ViewTreeObserver;
import android.view.WindowManager;  //
import android.util.DisplayMetrics;

import com.plattysoft.leonids.ParticleSystem;
import com.plattysoft.leonids.modifiers.AlphaModifier;
import com.plattysoft.leonids.modifiers.ScaleModifier;

import java.time.Clock;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
//        setSupportActionBar(toolbar);
//
//        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
//        fab.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
//                        .setAction("Action", null).show();
//            }
//        });
    }

//    @Override
//    public boolean onCreateOptionsMenu(Menu menu) {
//        // Inflate the menu; this adds items to the action bar if it is present.
//        getMenuInflater().inflate(R.menu.menu_main, menu);
//        return true;
//    }
//
//    @Override
//    public boolean onOptionsItemSelected(MenuItem item) {
//        // Handle action bar item clicks here. The action bar will
//        // automatically handle clicks on the Home/Up button, so long
//        // as you specify a parent activity in AndroidManifest.xml.
//        int id = item.getItemId();
//
//        //noinspection SimplifiableIfStatement
//        if (id == R.id.action_settings) {
//            return true;
//        }
//
//        return super.onOptionsItemSelected(item);
//    }

    @Override
    protected void onStart() {
        super.onStart();

//        ScrollView scroll = findViewById(R.id.scroll);
//        ParticleSystem ps = new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
//                .setSpeedRange(0.2f, 0.5f)
//                .oneShot(scroll, 500);
//        new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
//                .setSpeedRange(0.2f, 0.5f)
//                .oneShot(scroll, 500);


//        new ParticleSystem(this, 80, R.drawable.star_white, 1000)
//                .setSpeedModuleAndAngleRange(0f, 0.3f, 180, 180)
//                .setRotationSpeed(144)
//                .setAcceleration(0.00005f, 90)
//                .emit(findViewById(R.id.helloTextView), 8);

        Context context = getApplicationContext();
        final int totX = getScreenWidth(context);
        final int totY = getScreenHeight(context);
        int midX = totX / 2;
        int midY = totY / 2;



        final ParticleSystem ps = new ParticleSystem(this, 10, R.drawable.diamond, 2500)
                .setSpeedModuleAndAngleRange(0f, 0.03f, 90, 360)
//                            .setSpeedByComponentsRange(-0.1f, 0.1f, -0.1f, 0.02f)
//                            .setInitialRotationRange(0, 180)
//                            .setRotationSpeed(144)
//                            .setAcceleration(0.000001f, 90)
                .addModifier(new AlphaModifier(1, 2, 1, 1500))
                .addModifier(new ScaleModifier(1f, 2f, 1, 2000))
                .setFadeOut(1000);
        ps.emit(midX, midY, 5);

        /*  Method doesn't work well
        Timer myTimer = new Timer();
        myTimer.schedule(new TimerTask() {
            @Override
            public void run() {
                int emiterX = getRandomNumberInRange(600-1024/2, 600+1024/2);
                int emiterY = getRandomNumberInRange(600-768/2, 600+768/2);
                ps.updateEmitPoint(emiterX, emiterY);
                System.out.println("emiterX: " + emiterX + "\nemiterY: " + emiterY);
            }
        }, 5000);
        */





        /*  This block of code made emission from the View widget too fast
        final View emiter_center = findViewById(R.id.emiter_center);
        final ViewTreeObserver viewTreeObserver = emiter_center.getViewTreeObserver();
        if (viewTreeObserver.isAlive()) {
            viewTreeObserver.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
                @Override
                public void onGlobalLayout() {
//                    viewTreeObserver.removeOnGlobalLayoutListener();
                    new ParticleSystem(MainActivity.this, 2, R.drawable.star_white, 2500)
                            .setSpeedModuleAndAngleRange(0f, 0.1f, 0, 180)
//                            .setSpeedByComponentsRange(-0.1f, 0.1f, -0.1f, 0.02f)
//                            .setInitialRotationRange(0, 180)
//                            .setRotationSpeed(144)
//                            .setAcceleration(0.000001f, 90)
                            .addModifier(new AlphaModifier(1, 2, 1, 2000))
                            .addModifier(new ScaleModifier(3, 0.5f, 1, 2000))
                            .setFadeOut(1000)
                            .emit(emiter_center, 2, 100000);
                }
            });
        }

        */






//        new ParticleSystem(this, 80, R.drawable.star_white, 1000)
//                .setSpeedModuleAndAngleRange(0f, 0.3f, 180, 180)
//                .setRotationSpeed(144)
//                .setAcceleration(0.00005f, 90)
//                .emit(findViewById(R.id.emiter_top_right), 8);
//
//        new ParticleSystem(this, 80, R.drawable.star_white, 1000)
//                .setSpeedModuleAndAngleRange(0f, 0.3f, 0, 0)
//                .setRotationSpeed(144)
//                .setAcceleration(0.00005f, 90)
//                .emit(findViewById(R.id.emiter_top_left), 8);

        // NOTE: I got code from: https://guides.codepath.com/android/Repeating-Periodic-Tasks#executing-code-after-delay
        // Create the Handler object (on the main thread by default)
        final Handler handler = new Handler();
        // Define the code block to be executed
        Runnable runnableCode = new Runnable() {
            @Override
            public void run() {
                int emiterX = getRandomNumberInRange(0, totX);
                int emiterY = getRandomNumberInRange(0, totY);
                ps.updateEmitPoint(emiterX, emiterY);
//                System.out.println("emiterX: " + emiterX + "\nemiterY: " + emiterY);
                handler.postDelayed(this, 500);
            }
        };
        // Run the above code block on the main thread after 2 seconds
        handler.post(runnableCode);
    }





    // Custom method to get screen height /*in dp/dip using Context object*/
    public static int getScreenHeight(Context context){
        DisplayMetrics dm = new DisplayMetrics();
        WindowManager windowManager = (WindowManager) context.getSystemService(WINDOW_SERVICE);
        windowManager.getDefaultDisplay().getMetrics(dm);
        /*
            In this example code we converted the float value
            to nearest whole integer number. But, you can get the actual height in dp
            by removing the Math.round method. Then, it will return a float value, you should
            also make the necessary changes.
        */

        /*
            public int heightPixels
                The absolute height of the display in pixels.

            public float density
             The logical density of the display.
        int heightInDP = Math.round(dm.heightPixels / dm.density);
        return heightInDP;
        */
        return Math.round(dm.heightPixels);
    }

    // Custom method to get screen width /*in dp/dip using Context object*/
    public static int getScreenWidth(Context context){
        DisplayMetrics dm = new DisplayMetrics();
        WindowManager windowManager = (WindowManager) context.getSystemService(WINDOW_SERVICE);
        windowManager.getDefaultDisplay().getMetrics(dm);
        /*
            In this example code we converted the float value
            to nearest whole integer number. But, you can get the actual height in dp
            by removing the Math.round method. Then, it will return a float value, you should
            also make the necessary changes.
        */

        /*
            public int heightPixels
                The absolute height of the display in pixels.

            public float density
             The logical density of the display.
        int heightInDP = Math.round(dm.heightPixels / dm.density);
        return heightInDP;
        */
        return Math.round(dm.widthPixels);
    }




    private static int getRandomNumberInRange(int min, int max) {

        if (min >= max) {
            throw new IllegalArgumentException("max must be greater than min");
        }

        Random r = new Random();
        return r.nextInt((max - min) + 1) + min;
    }
}
