module class1_tree24(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[275]) & i[365]) | ((~i[293]) & (~i[365])) | (i[26] & i[293] & (~i[365])) | ((~i[398]) & i[360] & i[275] & i[365]) | ((~i[373]) & (~i[360]) & i[275] & i[365]) | (i[108] & i[398] & i[360] & i[275] & i[365]) | (i[147] & i[373] & (~i[360]) & i[275] & i[365]) | ((~i[147]) & i[373] & (~i[360]) & i[275] & i[365]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
