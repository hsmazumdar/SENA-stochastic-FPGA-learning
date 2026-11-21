module class7_tree14(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[460])) | ((~i[401]) & (~i[42]) & i[315] & i[211] & i[460]) | (i[429] & i[56] & (~i[315]) & i[211] & i[460]) | (i[191] & i[250] & i[293] & (~i[211]) & i[460]) | (i[238] & (~i[250]) & i[293] & (~i[211]) & i[460]) | (i[255] & i[66] & (~i[293]) & (~i[211]) & i[460]) | ((~i[185]) & (~i[66]) & (~i[293]) & (~i[211]) & i[460]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
