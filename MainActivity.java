package com.example.hungn.homescreen;

import android.content.Context;
import android.graphics.Typeface;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import com.plattysoft.leonids.ParticleSystem;
import com.plattysoft.leonids.modifiers.ScaleModifier;
import com.plattysoft.leonids.ParticleSystem;
import com.plattysoft.leonids.modifiers.AlphaModifier;
import com.plattysoft.leonids.modifiers.ScaleModifier;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
//import android.support.design.widget.FloatingActionButton;
//import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.view.ViewTreeObserver;
import android.view.WindowManager;  //
import android.util.DisplayMetrics;

import java.time.Clock;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    Button referenceButton;
    ImageButton accountinfo;
    TextView gameTitle;
    Typeface tf1;
    Typeface tf2;
    LinearLayout levelMenu;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        referenceButton = (Button) findViewById(R.id.referenceButton);
        tf1 = Typeface.createFromAsset(getAssets(), "font.ttf");

        accountinfo = (ImageButton) findViewById(R.id.accountinfo);

        gameTitle = (TextView) findViewById(R.id.title);
        referenceButton.setText("reference");
        referenceButton.setTypeface(tf1);
        gameTitle.setTypeface(tf1);

        referenceButton.getBackground().setAlpha(64);
        accountinfo.setAlpha(64);

        levelMenu = findViewById(R.id.levelMenu);
        /* Creating Level Menus*/

        ImageView backgroundLevel;

        View child1 = getLayoutInflater().inflate(R.layout.level, null);
        TextView title = child1.findViewById(R.id.textView);
        title.setText("Compass Heading");
        levelMenu.addView(child1);

        View child2 = getLayoutInflater().inflate(R.layout.level, null);
        title = child2.findViewById(R.id.textView);
        title.setText("Closest Heading");
        backgroundLevel = child2.findViewById(R.id.imageView);
        backgroundLevel.setBackgroundResource(R.drawable.level2);

        levelMenu.addView(child2);

        View child3 = getLayoutInflater().inflate(R.layout.level, null);
        title = child3.findViewById(R.id.textView);
        title.setText("Absolute Heading");
        backgroundLevel = child3.findViewById(R.id.imageView);
        backgroundLevel.setBackgroundResource(R.drawable.level3);
        levelMenu.addView(child3);

        View child4 = getLayoutInflater().inflate(R.layout.level, null);
        title = child4.findViewById(R.id.textView);
        title.setText("Simulated Aircraft");
        backgroundLevel = child4.findViewById(R.id.imageView);
        backgroundLevel.setBackgroundResource(R.drawable.level4);
        levelMenu.addView(child4);

        View child5 = getLayoutInflater().inflate(R.layout.level, null);
        title = child5.findViewById(R.id.textView);
        title.setText("Wind Effects");
        backgroundLevel = child5.findViewById(R.id.imageView);
        backgroundLevel.setBackgroundResource(R.drawable.level5);
        levelMenu.addView(child5);

        new ParticleSystem(this, 50, R.drawable.stareffect, 6000)
		.setSpeedByComponentsRange(-0.1f, 0.1f, -0.1f, 0.02f)
		.setAcceleration(0.000003f, 90)
		.setInitialRotationRange(0, 360)
		.setRotationSpeed(120)
		.setFadeOut(2000)
		.addModifier(new ScaleModifier(0f, 1.5f, 0, 1500))
//        .oneShot(findViewById(R.id.screen), 10)
//        .emitWithGravity(findViewById(R.id.screen), Gravity.BOTTOM, 30);
        .emit(findViewById(R.id.emiter_top_left), 100);

//        new ParticleSystem(this, 50, R.drawable.star, 1000, R.id.screenView)
//                .setSpeedRange(0.1f, 0.25f)
//                .emit(findViewById(R.id.screenView), 100);
    }

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
